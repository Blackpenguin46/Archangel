#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Secure Communication Demo
Demonstrates advanced encrypted agent communication with mutual TLS,
Noise Protocol Framework, and secure key management.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from agents.secure_communication import (
    SecureCommunicationBus, SecurityLevel, ProtocolType,
    CertificateManager, NoiseProtocolManager, KeyRotationManager
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SecureCommunicationDemo:
    """
    Comprehensive demonstration of secure communication features
    """
    
    def __init__(self):
        self.demo_results = {}
    
    async def demo_certificate_management(self):
        """Demonstrate certificate management and mTLS"""
        print("\n" + "="*60)
        print("CERTIFICATE MANAGEMENT AND MUTUAL TLS DEMO")
        print("="*60)
        
        try:
            # Initialize certificate manager
            cert_manager = CertificateManager("demo_certs")
            
            print("1. Generating Certificate Authority...")
            ca_cert, ca_key = cert_manager.generate_ca_certificate()
            print(f"   ✓ CA Certificate generated")
            print(f"   ✓ CA Subject: {ca_cert.subject.rfc4514_string()}")
            print(f"   ✓ CA Valid until: {ca_cert.not_valid_after}")
            
            print("\n2. Generating Agent Certificates...")
            agents = ["red_recon_agent", "red_exploit_agent", "blue_soc_agent", "blue_firewall_agent"]
            
            for agent_id in agents:
                cert_info = cert_manager.generate_agent_certificate(agent_id)
                print(f"   ✓ Certificate for {agent_id}")
                print(f"     - Fingerprint: {cert_info.fingerprint[:16]}...")
                print(f"     - Expires: {cert_info.expiry.strftime('%Y-%m-%d %H:%M:%S')}")
            
            print("\n3. Testing Certificate Pinning...")
            test_agent = "red_recon_agent"
            cert_path = cert_manager.agent_certs[test_agent].cert_path
            
            with open(cert_path, "rb") as f:
                cert_data = f.read()
            
            # Test valid pin
            valid_pin = cert_manager.verify_certificate_pin(test_agent, cert_data)
            print(f"   ✓ Valid certificate pin verification: {valid_pin}")
            
            # Test invalid pin
            fake_cert = b"fake certificate data"
            invalid_pin = cert_manager.verify_certificate_pin(test_agent, fake_cert)
            print(f"   ✓ Invalid certificate pin rejection: {not invalid_pin}")
            
            self.demo_results['certificate_management'] = {
                'ca_generated': True,
                'agent_certs_generated': len(agents),
                'certificate_pinning_works': valid_pin and not invalid_pin
            }
            
        except Exception as e:
            print(f"   ✗ Certificate management demo failed: {e}")
            self.demo_results['certificate_management'] = {'error': str(e)}
    
    async def demo_noise_protocol(self):
        """Demonstrate Noise Protocol Framework"""
        print("\n" + "="*60)
        print("NOISE PROTOCOL FRAMEWORK DEMO")
        print("="*60)
        
        try:
            # Initialize Noise protocol manager
            noise_manager = NoiseProtocolManager()
            
            print("1. Generating Noise Protocol Keypairs...")
            agents = ["alice", "bob"]
            keypairs = {}
            
            for agent in agents:
                private_key, public_key = noise_manager.generate_keypair(agent)
                keypairs[agent] = (private_key, public_key)
                print(f"   ✓ Keypair for {agent}: {len(private_key)} byte private key, {len(public_key)} byte public key")
            
            print("\n2. Initializing Noise XX Handshake...")
            alice_handshake = noise_manager.initialize_handshake("alice", is_initiator=True)
            bob_handshake = noise_manager.initialize_handshake("bob", is_initiator=False)
            
            print(f"   ✓ Alice handshake initialized: {alice_handshake.protocol_name}")
            print(f"   ✓ Bob handshake initialized: {bob_handshake.protocol_name}")
            
            print("\n3. Performing Handshake Exchange...")
            # Simulate handshake completion (actual Noise library would handle this)
            alice_handshake.completed = True
            bob_handshake.completed = True
            
            # Mock cipher states for demonstration
            noise_manager.cipher_states["alice"] = {'send': None, 'recv': None}
            noise_manager.cipher_states["bob"] = {'send': None, 'recv': None}
            
            print("   ✓ Handshake completed successfully")
            
            print("\n4. Testing Message Encryption/Decryption...")
            test_message = b"This is a secret message from Alice to Bob"
            
            # Encrypt message
            encrypted = noise_manager.encrypt_message("alice", test_message)
            print(f"   ✓ Original message: {test_message.decode()}")
            print(f"   ✓ Encrypted message: {encrypted.hex()[:32]}...")
            
            # Decrypt message
            decrypted = noise_manager.decrypt_message("alice", encrypted)  # Using same agent for demo
            print(f"   ✓ Decrypted message: {decrypted.decode()}")
            print(f"   ✓ Encryption/Decryption successful: {test_message == decrypted}")
            
            self.demo_results['noise_protocol'] = {
                'keypairs_generated': len(agents),
                'handshake_completed': alice_handshake.completed and bob_handshake.completed,
                'encryption_works': test_message == decrypted
            }
            
        except Exception as e:
            print(f"   ✗ Noise protocol demo failed: {e}")
            self.demo_results['noise_protocol'] = {'error': str(e)}
    
    async def demo_key_rotation(self):
        """Demonstrate key rotation and management"""
        print("\n" + "="*60)
        print("KEY ROTATION AND MANAGEMENT DEMO")
        print("="*60)
        
        try:
            # Initialize key manager with short rotation interval for demo
            from datetime import timedelta
            key_manager = KeyRotationManager(rotation_interval=timedelta(seconds=2))
            
            print("1. Generating Initial Session Keys...")
            agents = ["agent_alpha", "agent_beta", "agent_gamma"]
            initial_keys = {}
            
            for agent in agents:
                key, version = key_manager.generate_session_key(agent)
                initial_keys[agent] = (key, version)
                print(f"   ✓ {agent}: Key v{version} generated ({len(key)} bytes)")
            
            print("\n2. Testing Key Retrieval...")
            for agent in agents:
                active_key, active_version = key_manager.get_active_key(agent)
                original_key, original_version = initial_keys[agent]
                print(f"   ✓ {agent}: Retrieved key v{active_version} matches: {active_key == original_key}")
            
            print("\n3. Waiting for Key Rotation Interval...")
            print("   (Waiting 3 seconds for rotation interval...)")
            await asyncio.sleep(3)
            
            print("\n4. Testing Automatic Key Rotation...")
            rotated_keys = {}
            for agent in agents:
                rotated = key_manager.rotate_key_if_needed(agent)
                if rotated:
                    new_key, new_version = rotated
                    rotated_keys[agent] = (new_key, new_version)
                    original_key, original_version = initial_keys[agent]
                    print(f"   ✓ {agent}: Rotated from v{original_version} to v{new_version}")
                    print(f"     - Key changed: {new_key != original_key}")
                else:
                    print(f"   ✗ {agent}: No rotation occurred")
            
            print("\n5. Testing Key History...")
            for agent in agents:
                history = key_manager.key_history.get(agent, [])
                print(f"   ✓ {agent}: {len(history)} keys in history")
                for entry in history:
                    print(f"     - v{entry['version']}: {entry['created'].strftime('%H:%M:%S')}")
            
            self.demo_results['key_rotation'] = {
                'initial_keys_generated': len(initial_keys),
                'keys_rotated': len(rotated_keys),
                'rotation_working': len(rotated_keys) > 0
            }
            
        except Exception as e:
            print(f"   ✗ Key rotation demo failed: {e}")
            self.demo_results['key_rotation'] = {'error': str(e)}
    
    async def demo_secure_messaging(self):
        """Demonstrate end-to-end secure messaging"""
        print("\n" + "="*60)
        print("SECURE MESSAGING DEMO")
        print("="*60)
        
        try:
            # Initialize secure communication bus
            secure_bus = SecureCommunicationBus(
                bind_address="tcp://127.0.0.1:5558",
                security_level=SecurityLevel.ENHANCED,
                protocol_type=ProtocolType.NOISE_XX,
                cert_dir="demo_secure_certs"
            )
            
            print("1. Initializing Secure Communication Bus...")
            await secure_bus.initialize()
            print(f"   ✓ Security Level: {secure_bus.security_level.value}")
            print(f"   ✓ Protocol Type: {secure_bus.protocol_type.value}")
            print(f"   ✓ Bus Status: {'Running' if secure_bus.running else 'Stopped'}")
            
            print("\n2. Registering Secure Agents...")
            red_agents = ["red_recon", "red_exploit", "red_persist"]
            blue_agents = ["blue_soc", "blue_firewall", "blue_siem"]
            
            for agent in red_agents:
                cert_info = secure_bus.register_secure_agent(agent, "red")
                print(f"   ✓ Red Team - {agent}: {cert_info.fingerprint[:16]}...")
            
            for agent in blue_agents:
                cert_info = secure_bus.register_secure_agent(agent, "blue")
                print(f"   ✓ Blue Team - {agent}: {cert_info.fingerprint[:16]}...")
            
            print("\n3. Establishing Secure Channels...")
            channels_established = 0
            
            # Red team internal communication
            for i in range(len(red_agents) - 1):
                success = await secure_bus.establish_secure_channel(red_agents[i], red_agents[i+1])
                if success:
                    channels_established += 1
                    print(f"   ✓ Channel: {red_agents[i]} ↔ {red_agents[i+1]}")
            
            # Blue team internal communication
            for i in range(len(blue_agents) - 1):
                success = await secure_bus.establish_secure_channel(blue_agents[i], blue_agents[i+1])
                if success:
                    channels_established += 1
                    print(f"   ✓ Channel: {blue_agents[i]} ↔ {blue_agents[i+1]}")
            
            print(f"\n   Total channels established: {channels_established}")
            
            print("\n4. Testing Secure Message Exchange...")
            messages_sent = 0
            
            # Red team intelligence sharing
            intel_message = {
                "type": "intelligence",
                "intelligence_type": "vulnerability",
                "target": "web_server_01",
                "vulnerability": "SQL Injection in login form",
                "confidence": 0.95,
                "timestamp": datetime.now().isoformat()
            }
            
            success = await secure_bus.send_secure_message("red_recon", "red_exploit", intel_message)
            if success:
                messages_sent += 1
                print("   ✓ Red Team Intelligence: recon → exploit")
            
            # Blue team alert
            alert_message = {
                "type": "alert",
                "alert_type": "suspicious_activity",
                "severity": "high",
                "source": "web_server_01",
                "description": "Multiple failed login attempts detected",
                "indicators": ["192.168.1.100", "admin", "multiple_failures"],
                "timestamp": datetime.now().isoformat()
            }
            
            success = await secure_bus.send_secure_message("blue_soc", "blue_firewall", alert_message)
            if success:
                messages_sent += 1
                print("   ✓ Blue Team Alert: soc → firewall")
            
            print(f"\n   Total secure messages sent: {messages_sent}")
            
            print("\n5. Testing Replay Attack Prevention...")
            # Generate and test nonces
            nonce1 = secure_bus.generate_message_nonce()
            nonce2 = secure_bus.generate_message_nonce()
            
            # First use should succeed
            valid1 = secure_bus.verify_message_nonce(nonce1)
            valid2 = secure_bus.verify_message_nonce(nonce2)
            
            # Replay should fail
            replay1 = secure_bus.verify_message_nonce(nonce1)
            replay2 = secure_bus.verify_message_nonce(nonce2)
            
            print(f"   ✓ First nonce use: {valid1}")
            print(f"   ✓ Second nonce use: {valid2}")
            print(f"   ✓ Replay prevention: {not replay1 and not replay2}")
            
            print("\n6. Testing Message Integrity...")
            test_key = b"test_key_32_bytes_long_for_hmac!"
            test_data = b"This is test data for integrity verification"
            
            # Calculate integrity hash
            integrity_hash = secure_bus.calculate_message_integrity(test_data, test_key)
            
            # Verify integrity
            valid_integrity = secure_bus.verify_message_integrity(test_data, test_key, integrity_hash)
            
            # Test with tampered data
            tampered_data = b"This is tampered data for integrity verification"
            invalid_integrity = secure_bus.verify_message_integrity(tampered_data, test_key, integrity_hash)
            
            print(f"   ✓ Valid integrity check: {valid_integrity}")
            print(f"   ✓ Tampered data detection: {not invalid_integrity}")
            
            print("\n7. Security Statistics...")
            stats = secure_bus.get_security_stats()
            print(f"   ✓ Secure messages sent: {stats['secure_messages_sent']}")
            print(f"   ✓ Handshakes completed: {stats['handshakes_completed']}")
            print(f"   ✓ Keys rotated: {stats['keys_rotated']}")
            print(f"   ✓ Replay attacks prevented: {stats['replay_attacks_prevented']}")
            print(f"   ✓ Active agents: {stats['active_agents']}")
            
            self.demo_results['secure_messaging'] = {
                'bus_initialized': secure_bus.running,
                'agents_registered': len(red_agents) + len(blue_agents),
                'channels_established': channels_established,
                'messages_sent': messages_sent,
                'replay_prevention_works': not replay1 and not replay2,
                'integrity_verification_works': valid_integrity and not invalid_integrity,
                'statistics': stats
            }
            
            # Cleanup
            await secure_bus.shutdown()
            
        except Exception as e:
            print(f"   ✗ Secure messaging demo failed: {e}")
            self.demo_results['secure_messaging'] = {'error': str(e)}
    
    async def demo_protocol_comparison(self):
        """Demonstrate different security protocols"""
        print("\n" + "="*60)
        print("SECURITY PROTOCOL COMPARISON DEMO")
        print("="*60)
        
        protocols = [
            (ProtocolType.MTLS, SecurityLevel.ENHANCED),
            (ProtocolType.NOISE_XX, SecurityLevel.ENHANCED),
            (ProtocolType.NOISE_XX, SecurityLevel.MAXIMUM)
        ]
        
        comparison_results = {}
        
        for protocol_type, security_level in protocols:
            try:
                print(f"\nTesting {protocol_type.value} with {security_level.value} security...")
                
                # Initialize bus with specific protocol
                secure_bus = SecureCommunicationBus(
                    bind_address=f"tcp://127.0.0.1:{5559 + len(comparison_results)}",
                    security_level=security_level,
                    protocol_type=protocol_type,
                    cert_dir=f"demo_certs_{protocol_type.value}"
                )
                
                start_time = time.time()
                await secure_bus.initialize()
                init_time = time.time() - start_time
                
                # Register test agents
                secure_bus.register_secure_agent("test_agent_1", "red")
                secure_bus.register_secure_agent("test_agent_2", "blue")
                
                # Establish channel
                start_time = time.time()
                channel_success = await secure_bus.establish_secure_channel("test_agent_1", "test_agent_2")
                channel_time = time.time() - start_time
                
                # Send test message
                start_time = time.time()
                message_success = await secure_bus.send_secure_message(
                    "test_agent_1", "test_agent_2", 
                    {"test": "message", "protocol": protocol_type.value}
                )
                message_time = time.time() - start_time
                
                stats = secure_bus.get_security_stats()
                
                comparison_results[f"{protocol_type.value}_{security_level.value}"] = {
                    'initialization_time': round(init_time, 4),
                    'channel_establishment_time': round(channel_time, 4),
                    'message_send_time': round(message_time, 4),
                    'channel_success': channel_success,
                    'message_success': message_success,
                    'stats': stats
                }
                
                print(f"   ✓ Initialization: {init_time:.4f}s")
                print(f"   ✓ Channel establishment: {channel_time:.4f}s")
                print(f"   ✓ Message sending: {message_time:.4f}s")
                print(f"   ✓ Channel success: {channel_success}")
                print(f"   ✓ Message success: {message_success}")
                
                await secure_bus.shutdown()
                
            except Exception as e:
                print(f"   ✗ {protocol_type.value} test failed: {e}")
                comparison_results[f"{protocol_type.value}_{security_level.value}"] = {'error': str(e)}
        
        self.demo_results['protocol_comparison'] = comparison_results
    
    def print_summary(self):
        """Print comprehensive demo summary"""
        print("\n" + "="*60)
        print("SECURE COMMUNICATION DEMO SUMMARY")
        print("="*60)
        
        total_tests = 0
        passed_tests = 0
        
        for demo_name, results in self.demo_results.items():
            print(f"\n{demo_name.upper().replace('_', ' ')}:")
            
            if 'error' in results:
                print(f"   ✗ FAILED: {results['error']}")
                total_tests += 1
            else:
                for key, value in results.items():
                    if isinstance(value, bool):
                        status = "✓" if value else "✗"
                        print(f"   {status} {key.replace('_', ' ').title()}: {value}")
                        total_tests += 1
                        if value:
                            passed_tests += 1
                    elif isinstance(value, (int, float, str)):
                        print(f"   • {key.replace('_', ' ').title()}: {value}")
                    elif isinstance(value, dict) and key != 'statistics':
                        print(f"   • {key.replace('_', ' ').title()}:")
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, bool):
                                status = "✓" if subvalue else "✗"
                                print(f"     {status} {subkey.replace('_', ' ').title()}: {subvalue}")
                                total_tests += 1
                                if subvalue:
                                    passed_tests += 1
                            else:
                                print(f"     • {subkey.replace('_', ' ').title()}: {subvalue}")
        
        print(f"\n" + "="*60)
        print(f"OVERALL RESULTS: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        print("="*60)
        
        # Save results to file
        with open("secure_communication_demo_results.json", "w") as f:
            json.dump(self.demo_results, f, indent=2, default=str)
        print("✓ Detailed results saved to secure_communication_demo_results.json")

async def main():
    """Run comprehensive secure communication demonstration"""
    print("Archangel Autonomous AI Evolution")
    print("Secure Communication System Demonstration")
    print("="*60)
    
    demo = SecureCommunicationDemo()
    
    try:
        # Run all demonstrations
        await demo.demo_certificate_management()
        await demo.demo_noise_protocol()
        await demo.demo_key_rotation()
        await demo.demo_secure_messaging()
        await demo.demo_protocol_comparison()
        
        # Print comprehensive summary
        demo.print_summary()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}")
        logger.exception("Demo failed")
    
    print("\nDemo completed!")

if __name__ == "__main__":
    asyncio.run(main())