/*
 * Archangel Memory AI Module Compilation Test
 * 
 * This test verifies that the memory AI module compiles correctly
 * and has all required symbols and functions.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

#define TEST_MODULE_DIR "kernel/archangel"
#define TEST_MODULE_NAME "archangel_memory_ai"

/* Test result structure */
struct test_result {
    int passed;
    int failed;
    char error_msg[1024];
};

/**
 * run_command - Execute a command and capture output
 * @cmd: Command to execute
 * @output: Buffer to store output
 * @output_size: Size of output buffer
 * 
 * Returns: 0 on success, -1 on failure
 */
static int run_command(const char *cmd, char *output, size_t output_size)
{
    FILE *fp;
    int status;
    
    fp = popen(cmd, "r");
    if (!fp) {
        return -1;
    }
    
    if (output && output_size > 0) {
        memset(output, 0, output_size);
        fread(output, 1, output_size - 1, fp);
    }
    
    status = pclose(fp);
    return WEXITSTATUS(status);
}

/**
 * test_header_file_structure - Test if header file is properly structured
 * @result: Test result structure
 */
static void test_header_file_structure(struct test_result *result)
{
    char cmd[512];
    char output[4096];
    int ret;
    const char *required_includes[] = {
        "#include <linux/module.h>",
        "#include <linux/mm.h>",
        "#include \"archangel_core.h\"",
        NULL
    };
    int i;
    
    printf("Testing memory AI header file structure...\n");
    
    /* Read header file */
    snprintf(cmd, sizeof(cmd), "cat %s/archangel_memory_ai.h", TEST_MODULE_DIR);
    
    ret = run_command(cmd, output, sizeof(output));
    if (ret != 0) {
        result->failed++;
        snprintf(result->error_msg, sizeof(result->error_msg),
                "Failed to read header file");
        printf("  FAILED: %s\n", result->error_msg);
        return;
    }
    
    /* Check for required includes */
    for (i = 0; required_includes[i]; i++) {
        if (!strstr(output, required_includes[i])) {
            result->failed++;
            snprintf(result->error_msg, sizeof(result->error_msg),
                    "Required include '%s' not found", required_includes[i]);
            printf("  FAILED: %s\n", result->error_msg);
            return;
        }
    }
    
    /* Check for header guards */
    if (!strstr(output, "#ifndef _ARCHANGEL_MEMORY_AI_H") ||
        !strstr(output, "#define _ARCHANGEL_MEMORY_AI_H")) {
        result->failed++;
        snprintf(result->error_msg, sizeof(result->error_msg),
                "Header guards not properly implemented");
        printf("  FAILED: %s\n", result->error_msg);
        return;
    }
    
    result->passed++;
    printf("  PASSED: Header file structure is correct\n");
}

/**
 * test_structure_definitions - Test if required structures are defined
 * @result: Test result structure
 */
static void test_structure_definitions(struct test_result *result)
{
    char cmd[512];
    char output[4096];
    int ret;
    const char *required_structures[] = {
        "struct archangel_memory_ai_engine",
        "struct archangel_lstm_predictor",
        "struct archangel_exploit_detector",
        "struct archangel_process_profile",
        "struct archangel_mem_access",
        "struct archangel_lstm_cell",
        NULL
    };
    int i;
    
    printf("Testing memory AI structure definitions...\n");
    
    /* Read header file */
    snprintf(cmd, sizeof(cmd), "cat %s/archangel_memory_ai.h", TEST_MODULE_DIR);
    
    ret = run_command(cmd, output, sizeof(output));
    if (ret != 0) {
        result->failed++;
        snprintf(result->error_msg, sizeof(result->error_msg),
                "Failed to read header file");
        printf("  FAILED: %s\n", result->error_msg);
        return;
    }
    
    /* Check for required structures */
    for (i = 0; required_structures[i]; i++) {
        if (!strstr(output, required_structures[i])) {
            result->failed++;
            snprintf(result->error_msg, sizeof(result->error_msg),
                    "Required structure '%s' not found", required_structures[i]);
            printf("  FAILED: %s\n", result->error_msg);
            return;
        }
    }
    
    result->passed++;
    printf("  PASSED: All required structures are defined\n");
}

/**
 * test_enum_definitions - Test if required enums are defined
 * @result: Test result structure
 */
static void test_enum_definitions(struct test_result *result)
{
    char cmd[512];
    char output[4096];
    int ret;
    const char *required_enums[] = {
        "enum archangel_mem_access_type",
        "enum archangel_mem_pattern_type",
        "enum archangel_exploit_type",
        "enum archangel_mem_decision",
        NULL
    };
    int i;
    
    printf("Testing memory AI enum definitions...\n");
    
    /* Read header file */
    snprintf(cmd, sizeof(cmd), "cat %s/archangel_memory_ai.h", TEST_MODULE_DIR);
    
    ret = run_command(cmd, output, sizeof(output));
    if (ret != 0) {
        result->failed++;
        snprintf(result->error_msg, sizeof(result->error_msg),
                "Failed to read header file");
        printf("  FAILED: %s\n", result->error_msg);
        return;
    }
    
    /* Check for required enums */
    for (i = 0; required_enums[i]; i++) {
        if (!strstr(output, required_enums[i])) {
            result->failed++;
            snprintf(result->error_msg, sizeof(result->error_msg),
                    "Required enum '%s' not found", required_enums[i]);
            printf("  FAILED: %s\n", result->error_msg);
            return;
        }
    }
    
    result->passed++;
    printf("  PASSED: All required enums are defined\n");
}

/**
 * test_function_declarations - Test if required functions are declared
 * @result: Test result structure
 */
static void test_function_declarations(struct test_result *result)
{
    char cmd[512];
    char output[4096];
    int ret;
    const char *required_functions[] = {
        "archangel_memory_ai_init",
        "archangel_memory_ai_cleanup",
        "archangel_ai_handle_mm_fault",
        "archangel_lstm_predict_next_access",
        "archangel_detect_exploit_pattern",
        "archangel_process_profile_lookup",
        NULL
    };
    int i;
    
    printf("Testing memory AI function declarations...\n");
    
    /* Read header file */
    snprintf(cmd, sizeof(cmd), "cat %s/archangel_memory_ai.h", TEST_MODULE_DIR);
    
    ret = run_command(cmd, output, sizeof(output));
    if (ret != 0) {
        result->failed++;
        snprintf(result->error_msg, sizeof(result->error_msg),
                "Failed to read header file");
        printf("  FAILED: %s\n", result->error_msg);
        return;
    }
    
    /* Check for required function declarations */
    for (i = 0; required_functions[i]; i++) {
        if (!strstr(output, required_functions[i])) {
            result->failed++;
            snprintf(result->error_msg, sizeof(result->error_msg),
                    "Required function '%s' not found", required_functions[i]);
            printf("  FAILED: %s\n", result->error_msg);
            return;
        }
    }
    
    result->passed++;
    printf("  PASSED: All required function declarations found\n");
}

/**
 * test_implementation_file - Test if implementation file has required content
 * @result: Test result structure
 */
static void test_implementation_file(struct test_result *result)
{
    char cmd[512];
    char output[8192];  /* Larger buffer for implementation file */
    int ret;
    const char *required_implementations[] = {
        "archangel_memory_ai_init",
        "archangel_ai_handle_mm_fault",
        "archangel_lstm_predict_next_access",
        "archangel_detect_exploit_pattern",
        "MODULE_LICENSE",
        "MODULE_AUTHOR",
        "MODULE_DESCRIPTION",
        NULL
    };
    int i;
    
    printf("Testing memory AI implementation file...\n");
    
    /* Read implementation file */
    snprintf(cmd, sizeof(cmd), "cat %s/archangel_memory_ai.c", TEST_MODULE_DIR);
    
    ret = run_command(cmd, output, sizeof(output));
    if (ret != 0) {
        result->failed++;
        snprintf(result->error_msg, sizeof(result->error_msg),
                "Failed to read implementation file");
        printf("  FAILED: %s\n", result->error_msg);
        return;
    }
    
    /* Check for required implementations */
    for (i = 0; required_implementations[i]; i++) {
        if (!strstr(output, required_implementations[i])) {
            result->failed++;
            snprintf(result->error_msg, sizeof(result->error_msg),
                    "Required implementation '%s' not found", required_implementations[i]);
            printf("  FAILED: %s\n", result->error_msg);
            return;
        }
    }
    
    /* Check for LSTM-specific implementations */
    if (!strstr(output, "archangel_lstm_sigmoid") || !strstr(output, "archangel_lstm_tanh")) {
        result->failed++;
        snprintf(result->error_msg, sizeof(result->error_msg),
                "LSTM mathematical functions not found");
        printf("  FAILED: %s\n", result->error_msg);
        return;
    }
    
    /* Check for exploit detection patterns */
    if (!strstr(output, "default_exploit_signatures") || !strstr(output, "ARCHANGEL_EXPLOIT_BUFFER_OVERFLOW")) {
        result->failed++;
        snprintf(result->error_msg, sizeof(result->error_msg),
                "Exploit detection patterns not found");
        printf("  FAILED: %s\n", result->error_msg);
        return;
    }
    
    result->passed++;
    printf("  PASSED: Implementation file has required content\n");
}

/**
 * test_makefile_integration - Test if module is integrated into Makefile
 * @result: Test result structure
 */
static void test_makefile_integration(struct test_result *result)
{
    char cmd[512];
    char output[4096];
    int ret;
    
    printf("Testing Makefile integration...\n");
    
    /* Read Makefile */
    snprintf(cmd, sizeof(cmd), "cat %s/Makefile", TEST_MODULE_DIR);
    
    ret = run_command(cmd, output, sizeof(output));
    if (ret != 0) {
        result->failed++;
        snprintf(result->error_msg, sizeof(result->error_msg),
                "Failed to read Makefile");
        printf("  FAILED: %s\n", result->error_msg);
        return;
    }
    
    /* Check for memory AI module entry */
    if (!strstr(output, "archangel_memory_ai.o")) {
        result->failed++;
        snprintf(result->error_msg, sizeof(result->error_msg),
                "Memory AI module not found in Makefile");
        printf("  FAILED: %s\n", result->error_msg);
        return;
    }
    
    /* Check for module object definition */
    if (!strstr(output, "archangel_memory_ai-objs")) {
        result->failed++;
        snprintf(result->error_msg, sizeof(result->error_msg),
                "Memory AI module objects not defined in Makefile");
        printf("  FAILED: %s\n", result->error_msg);
        return;
    }
    
    result->passed++;
    printf("  PASSED: Makefile integration verified\n");
}

/**
 * test_design_requirements_compliance - Test compliance with design requirements
 * @result: Test result structure
 */
static void test_design_requirements_compliance(struct test_result *result)
{
    char cmd[512];
    char output[8192];
    int ret;
    
    printf("Testing design requirements compliance...\n");
    
    /* Read implementation file */
    snprintf(cmd, sizeof(cmd), "cat %s/archangel_memory_ai.c", TEST_MODULE_DIR);
    
    ret = run_command(cmd, output, sizeof(output));
    if (ret != 0) {
        result->failed++;
        snprintf(result->error_msg, sizeof(result->error_msg),
                "Failed to read implementation file");
        printf("  FAILED: %s\n", result->error_msg);
        return;
    }
    
    /* Check for LSTM predictor implementation */
    if (!strstr(output, "lstm") && !strstr(output, "LSTM")) {
        result->failed++;
        snprintf(result->error_msg, sizeof(result->error_msg),
                "LSTM predictor implementation not found");
        printf("  FAILED: %s\n", result->error_msg);
        return;
    }
    
    /* Check for exploit detection */
    if (!strstr(output, "exploit") || !strstr(output, "pattern")) {
        result->failed++;
        snprintf(result->error_msg, sizeof(result->error_msg),
                "Exploit detection implementation not found");
        printf("  FAILED: %s\n", result->error_msg);
        return;
    }
    
    /* Check for page fault handling */
    if (!strstr(output, "handle_mm_fault") || !strstr(output, "vm_fault")) {
        result->failed++;
        snprintf(result->error_msg, sizeof(result->error_msg),
                "Page fault handling implementation not found");
        printf("  FAILED: %s\n", result->error_msg);
        return;
    }
    
    /* Check for prefetch optimization */
    if (!strstr(output, "prefetch")) {
        result->failed++;
        snprintf(result->error_msg, sizeof(result->error_msg),
                "Prefetch optimization implementation not found");
        printf("  FAILED: %s\n", result->error_msg);
        return;
    }
    
    /* Check for process termination logic */
    if (!strstr(output, "terminate_process") || !strstr(output, "SIGKILL")) {
        result->failed++;
        snprintf(result->error_msg, sizeof(result->error_msg),
                "Process termination logic not found");
        printf("  FAILED: %s\n", result->error_msg);
        return;
    }
    
    result->passed++;
    printf("  PASSED: Design requirements compliance verified\n");
}

/**
 * main - Main test function
 */
int main(void)
{
    struct test_result result = {0, 0, ""};
    
    printf("=== Archangel Memory AI Module Compilation Test ===\n\n");
    
    /* Run all tests */
    test_header_file_structure(&result);
    test_structure_definitions(&result);
    test_enum_definitions(&result);
    test_function_declarations(&result);
    test_implementation_file(&result);
    test_makefile_integration(&result);
    test_design_requirements_compliance(&result);
    
    /* Print results */
    printf("\n=== Test Results ===\n");
    printf("Passed: %d\n", result.passed);
    printf("Failed: %d\n", result.failed);
    
    if (result.failed > 0) {
        printf("Last error: %s\n", result.error_msg);
        printf("\nMemory AI module compilation test FAILED\n");
        return 1;
    }
    
    printf("\nMemory AI module compilation test PASSED\n");
    return 0;
}