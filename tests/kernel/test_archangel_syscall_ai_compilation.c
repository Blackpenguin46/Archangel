/*
 * Archangel Linux - Syscall AI Module Compilation Test
 * 
 * This test verifies that the syscall AI module compiles correctly
 * and basic functionality works as expected.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <string.h>
#include <errno.h>

/* Test configuration */
#define TEST_MODULE_NAME "archangel_syscall_ai"
#define TEST_PROC_PATH "/proc/archangel/stats"
#define TEST_TIMEOUT 30

/* Test result structure */
struct test_result {
    int passed;
    int failed;
    char error_msg[256];
};

/**
 * run_command - Execute a shell command and return exit status
 * @cmd: Command to execute
 * @output: Buffer to store command output (can be NULL)
 * @output_size: Size of output buffer
 * 
 * Returns: Command exit status
 */
int run_command(const char *cmd, char *output, size_t output_size)
{
    FILE *fp;
    int status;
    
    printf("Executing: %s\n", cmd);
    
    if (output) {
        fp = popen(cmd, "r");
        if (!fp) {
            printf("ERROR: Failed to execute command: %s\n", strerror(errno));
            return -1;
        }
        
        if (fgets(output, output_size, fp) == NULL) {
            output[0] = '\0';
        }
        
        status = pclose(fp);
    } else {
        status = system(cmd);
    }
    
    return WEXITSTATUS(status);
}

/**
 * test_module_compilation - Test if the module compiles successfully
 * @result: Test result structure
 */
void test_module_compilation(struct test_result *result)
{
    int ret;
    char output[1024];
    
    printf("\n=== Testing Syscall AI Module Compilation ===\n");
    
    /* Clean previous builds */
    ret = run_command("cd kernel/archangel && make clean", NULL, 0);
    if (ret != 0) {
        printf("WARNING: Clean command failed (this is usually OK)\n");
    }
    
    /* Attempt to build the module */
    ret = run_command("cd kernel/archangel && make modules 2>&1", output, sizeof(output));
    if (ret == 0) {
        printf("✓ Module compilation successful\n");
        result->passed++;
    } else {
        printf("✗ Module compilation failed\n");
        printf("Build output: %s\n", output);
        snprintf(result->error_msg, sizeof(result->error_msg), 
                "Module compilation failed with exit code %d", ret);
        result->failed++;
        return;
    }
    
    /* Check if module files were created */
    ret = run_command("ls kernel/archangel/" TEST_MODULE_NAME ".ko", NULL, 0);
    if (ret == 0) {
        printf("✓ Module file created successfully\n");
        result->passed++;
    } else {
        printf("✗ Module file not found\n");
        snprintf(result->error_msg, sizeof(result->error_msg), 
                "Module file " TEST_MODULE_NAME ".ko not created");
        result->failed++;
    }
}

/**
 * test_module_info - Test module information
 * @result: Test result structure
 */
void test_module_info(struct test_result *result)
{
    int ret;
    char output[1024];
    
    printf("\n=== Testing Module Information ===\n");
    
    /* Check module info */
    ret = run_command("cd kernel/archangel && modinfo " TEST_MODULE_NAME ".ko", 
                     output, sizeof(output));
    if (ret == 0) {
        printf("✓ Module info accessible\n");
        printf("Module info: %s\n", output);
        result->passed++;
    } else {
        printf("✗ Module info failed\n");
        snprintf(result->error_msg, sizeof(result->error_msg), 
                "Failed to get module info");
        result->failed++;
    }
    
    /* Check for required symbols */
    ret = run_command("cd kernel/archangel && nm " TEST_MODULE_NAME ".ko | grep -E '(ai_syscall_intercept|archangel_syscall_ai_init)'", 
                     output, sizeof(output));
    if (ret == 0) {
        printf("✓ Required symbols found in module\n");
        result->passed++;
    } else {
        printf("✗ Required symbols not found\n");
        snprintf(result->error_msg, sizeof(result->error_msg), 
                "Required symbols missing from module");
        result->failed++;
    }
}

/**
 * test_header_includes - Test header file includes and dependencies
 * @result: Test result structure
 */
void test_header_includes(struct test_result *result)
{
    int ret;
    
    printf("\n=== Testing Header Dependencies ===\n");
    
    /* Test header file syntax */
    ret = run_command("cd kernel/archangel && cpp -I/lib/modules/$(uname -r)/build/include "
                     "-I/lib/modules/$(uname -r)/build/arch/x86/include "
                     "archangel_syscall_ai.h > /dev/null 2>&1", NULL, 0);
    if (ret == 0) {
        printf("✓ Header file syntax valid\n");
        result->passed++;
    } else {
        printf("✗ Header file syntax errors\n");
        snprintf(result->error_msg, sizeof(result->error_msg), 
                "Header file contains syntax errors");
        result->failed++;
    }
    
    /* Check for required includes */
    ret = run_command("grep -q '#include.*archangel_core.h' kernel/archangel/archangel_syscall_ai.h", NULL, 0);
    if (ret == 0) {
        printf("✓ Core header included\n");
        result->passed++;
    } else {
        printf("✗ Core header not included\n");
        snprintf(result->error_msg, sizeof(result->error_msg), 
                "archangel_core.h not included in syscall AI header");
        result->failed++;
    }
}

/**
 * test_structure_definitions - Test structure definitions
 * @result: Test result structure
 */
void test_structure_definitions(struct test_result *result)
{
    int ret;
    
    printf("\n=== Testing Structure Definitions ===\n");
    
    /* Check for key structures */
    ret = run_command("grep -q 'struct archangel_syscall_ai_engine' kernel/archangel/archangel_syscall_ai.h", NULL, 0);
    if (ret == 0) {
        printf("✓ Syscall AI engine structure defined\n");
        result->passed++;
    } else {
        printf("✗ Syscall AI engine structure missing\n");
        snprintf(result->error_msg, sizeof(result->error_msg), 
                "archangel_syscall_ai_engine structure not defined");
        result->failed++;
    }
    
    ret = run_command("grep -q 'struct archangel_process_profile' kernel/archangel/archangel_syscall_ai.h", NULL, 0);
    if (ret == 0) {
        printf("✓ Process profile structure defined\n");
        result->passed++;
    } else {
        printf("✗ Process profile structure missing\n");
        snprintf(result->error_msg, sizeof(result->error_msg), 
                "archangel_process_profile structure not defined");
        result->failed++;
    }
    
    ret = run_command("grep -q 'struct archangel_decision_cache' kernel/archangel/archangel_syscall_ai.h", NULL, 0);
    if (ret == 0) {
        printf("✓ Decision cache structure defined\n");
        result->passed++;
    } else {
        printf("✗ Decision cache structure missing\n");
        snprintf(result->error_msg, sizeof(result->error_msg), 
                "archangel_decision_cache structure not defined");
        result->failed++;
    }
}

/**
 * test_function_exports - Test function exports and symbols
 * @result: Test result structure
 */
void test_function_exports(struct test_result *result)
{
    int ret;
    
    printf("\n=== Testing Function Exports ===\n");
    
    /* Check for main intercept function */
    ret = run_command("grep -q 'ai_syscall_intercept' kernel/archangel/archangel_syscall_ai.c", NULL, 0);
    if (ret == 0) {
        printf("✓ Main intercept function implemented\n");
        result->passed++;
    } else {
        printf("✗ Main intercept function missing\n");
        snprintf(result->error_msg, sizeof(result->error_msg), 
                "ai_syscall_intercept function not implemented");
        result->failed++;
    }
    
    /* Check for pattern matching */
    ret = run_command("grep -q 'archangel_pattern_match' kernel/archangel/archangel_syscall_ai.c", NULL, 0);
    if (ret == 0) {
        printf("✓ Pattern matching function implemented\n");
        result->passed++;
    } else {
        printf("✗ Pattern matching function missing\n");
        snprintf(result->error_msg, sizeof(result->error_msg), 
                "archangel_pattern_match function not implemented");
        result->failed++;
    }
    
    /* Check for decision caching */
    ret = run_command("grep -q 'archangel_decision_cache_lookup' kernel/archangel/archangel_syscall_ai.c", NULL, 0);
    if (ret == 0) {
        printf("✓ Decision caching function implemented\n");
        result->passed++;
    } else {
        printf("✗ Decision caching function missing\n");
        snprintf(result->error_msg, sizeof(result->error_msg), 
                "archangel_decision_cache_lookup function not implemented");
        result->failed++;
    }
}

/**
 * main - Main test function
 */
int main(void)
{
    struct test_result result = {0, 0, ""};
    
    printf("Archangel Linux - Syscall AI Module Compilation Test\n");
    printf("====================================================\n");
    
    /* Run all tests */
    test_module_compilation(&result);
    test_module_info(&result);
    test_header_includes(&result);
    test_structure_definitions(&result);
    test_function_exports(&result);
    
    /* Print results */
    printf("\n=== Test Results ===\n");
    printf("Tests passed: %d\n", result.passed);
    printf("Tests failed: %d\n", result.failed);
    
    if (result.failed > 0) {
        printf("Last error: %s\n", result.error_msg);
        printf("\n✗ SYSCALL AI MODULE COMPILATION TEST FAILED\n");
        return 1;
    } else {
        printf("\n✓ SYSCALL AI MODULE COMPILATION TEST PASSED\n");
        return 0;
    }
}