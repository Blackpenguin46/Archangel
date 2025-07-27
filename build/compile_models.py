#!/usr/bin/env python3
"""
Archangel Linux AI Model Compilation Script

This script converts AI models to formats suitable for kernel integration,
primarily focusing on TensorFlow Lite conversion for lightweight inference.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_tensorflow():
    """Import and configure TensorFlow with appropriate settings."""
    try:
        import tensorflow as tf
        # Suppress TensorFlow warnings for cleaner output
        tf.get_logger().setLevel('ERROR')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        return tf
    except ImportError:
        logger.error("TensorFlow not found. Please install: pip install tensorflow")
        sys.exit(1)

def convert_to_tflite(model_path, output_path, optimize=True):
    """
    Convert a TensorFlow model to TensorFlow Lite format.
    
    Args:
        model_path (str): Path to the input model
        output_path (str): Path for the output .tflite file
        optimize (bool): Whether to apply optimizations
    
    Returns:
        bool: True if conversion successful, False otherwise
    """
    tf = setup_tensorflow()
    
    try:
        logger.info(f"Converting {model_path} to TensorFlow Lite...")
        
        # Load the model
        if model_path.endswith('.h5'):
            model = tf.keras.models.load_model(model_path)
        elif model_path.endswith('.pb') or os.path.isdir(model_path):
            model = tf.saved_model.load(model_path)
        else:
            logger.error(f"Unsupported model format: {model_path}")
            return False
        
        # Create TensorFlow Lite converter
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path) if os.path.isdir(model_path) else tf.lite.TFLiteConverter.from_keras_model(model)
        
        if optimize:
            # Apply optimizations for kernel deployment
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Use integer quantization for better performance
            converter.representative_dataset = None  # Would need actual data for full quantization
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        
        # Convert the model
        tflite_model = converter.convert()
        
        # Save the converted model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Get model size information
        original_size = os.path.getsize(model_path) if os.path.isfile(model_path) else sum(os.path.getsize(os.path.join(dirpath, filename)) for dirpath, dirnames, filenames in os.walk(model_path) for filename in filenames)
        tflite_size = os.path.getsize(output_path)
        
        logger.info(f"Conversion successful!")
        logger.info(f"Original size: {original_size / 1024:.1f} KB")
        logger.info(f"TFLite size: {tflite_size / 1024:.1f} KB")
        logger.info(f"Compression ratio: {original_size / tflite_size:.1f}x")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to convert {model_path}: {str(e)}")
        return False

def generate_kernel_header(tflite_path, header_path, model_name):
    """
    Generate a C header file containing the TensorFlow Lite model as a byte array.
    
    Args:
        tflite_path (str): Path to the .tflite file
        header_path (str): Path for the output .h file
        model_name (str): Name for the model variable
    """
    try:
        logger.info(f"Generating kernel header for {model_name}...")
        
        with open(tflite_path, 'rb') as f:
            model_data = f.read()
        
        header_content = f"""/* Auto-generated header for {model_name} */
#ifndef _ARCHANGEL_{model_name.upper()}_MODEL_H
#define _ARCHANGEL_{model_name.upper()}_MODEL_H

#include <linux/types.h>

/* Model size and data */
#define ARCHANGEL_{model_name.upper()}_MODEL_SIZE {len(model_data)}

static const u8 archangel_{model_name}_model_data[] = {{
"""
        
        # Convert bytes to C array format
        for i, byte in enumerate(model_data):
            if i % 12 == 0:
                header_content += "\n    "
            header_content += f"0x{byte:02x}, "
        
        header_content += f"""
}};

#endif /* _ARCHANGEL_{model_name.upper()}_MODEL_H */
"""
        
        with open(header_path, 'w') as f:
            f.write(header_content)
        
        logger.info(f"Header generated: {header_path}")
        logger.info(f"Model size: {len(model_data)} bytes")
        
    except Exception as e:
        logger.error(f"Failed to generate header for {model_name}: {str(e)}")

def create_placeholder_models(output_dir):
    """
    Create placeholder model files for development when no real models exist.
    
    Args:
        output_dir (str): Output directory for placeholder models
    """
    logger.info("Creating placeholder models for development...")
    
    # Placeholder model definitions
    models = {
        'syscall_filter': {
            'description': 'Syscall filtering decision tree',
            'input_size': 64,
            'output_size': 1
        },
        'network_classifier': {
            'description': 'Network packet classification',
            'input_size': 128,
            'output_size': 10
        },
        'memory_analyzer': {
            'description': 'Memory pattern analysis',
            'input_size': 256,
            'output_size': 5
        }
    }
    
    for model_name, config in models.items():
        header_path = os.path.join(output_dir, f"{model_name}_model.h")
        
        # Create a minimal placeholder header
        header_content = f"""/* Placeholder header for {model_name} */
#ifndef _ARCHANGEL_{model_name.upper()}_MODEL_H
#define _ARCHANGEL_{model_name.upper()}_MODEL_H

#include <linux/types.h>

/* {config['description']} */
#define ARCHANGEL_{model_name.upper()}_INPUT_SIZE {config['input_size']}
#define ARCHANGEL_{model_name.upper()}_OUTPUT_SIZE {config['output_size']}
#define ARCHANGEL_{model_name.upper()}_MODEL_SIZE 1024

/* Placeholder model data - replace with actual compiled model */
static const u8 archangel_{model_name}_model_data[ARCHANGEL_{model_name.upper()}_MODEL_SIZE] = {{
    /* Placeholder data - all zeros for now */
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    /* ... (rest filled with zeros) ... */
}};

/* Placeholder inference function */
static inline int archangel_{model_name}_inference(const u8 *input, u8 *output)
{{
    /* Placeholder implementation - always returns 0 */
    if (!input || !output)
        return -1;
    
    /* Simple placeholder logic */
    output[0] = input[0] > 128 ? 1 : 0;
    return 0;
}}

#endif /* _ARCHANGEL_{model_name.upper()}_MODEL_H */
"""
        
        with open(header_path, 'w') as f:
            f.write(header_content)
        
        logger.info(f"Created placeholder model: {header_path}")

def main():
    parser = argparse.ArgumentParser(description='Compile AI models for Archangel kernel integration')
    parser.add_argument('--input', '-i', required=True, help='Input directory containing AI models')
    parser.add_argument('--output', '-o', required=True, help='Output directory for compiled models')
    parser.add_argument('--optimize', action='store_true', default=True, help='Apply optimizations (default: True)')
    parser.add_argument('--placeholder', action='store_true', help='Create placeholder models for development')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Archangel AI Model Compilation")
    logger.info(f"Input directory: {args.input}")
    logger.info(f"Output directory: {args.output}")
    
    # Check if input directory exists
    input_dir = Path(args.input)
    if not input_dir.exists() or args.placeholder:
        logger.warning(f"Input directory not found or placeholder requested")
        create_placeholder_models(args.output)
        return
    
    # Find and convert models
    model_files = []
    for ext in ['*.h5', '*.pb']:
        model_files.extend(input_dir.glob(ext))
    
    # Also check for SavedModel directories
    for item in input_dir.iterdir():
        if item.is_dir() and (item / 'saved_model.pb').exists():
            model_files.append(item)
    
    if not model_files:
        logger.warning("No models found in input directory")
        create_placeholder_models(args.output)
        return
    
    success_count = 0
    for model_path in model_files:
        model_name = model_path.stem
        tflite_path = output_dir / f"{model_name}.tflite"
        header_path = output_dir / f"{model_name}_model.h"
        
        # Convert to TensorFlow Lite
        if convert_to_tflite(str(model_path), str(tflite_path), args.optimize):
            # Generate kernel header
            generate_kernel_header(str(tflite_path), str(header_path), model_name)
            success_count += 1
    
    logger.info(f"Compilation completed: {success_count}/{len(model_files)} models converted successfully")

if __name__ == '__main__':
    main()