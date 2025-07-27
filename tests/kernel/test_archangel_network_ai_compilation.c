/*
 * Archangel Network AI Module Compilation Test
 * 
 * This test verifies that the network AI module compiles correctly
 * and has all required symbols and functions.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

#define TEST_MODULE_DIR "kernel/archangel"
#define TEST_MODULE_NAME "archangel_network_ai"

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
 * test_module_compilation - Test if the module compiles
 * @result: Test result structure
 */
static void test_module_compilation(struct test_result *result)
{
    char cmd[512];
    char output[4096];
    int ret;
    
    printf("Testing network AI module compilation...\n");
    
    /* Change to module directory and compile */
    snprintf(cmd, sizeof(cmd), "cd %s && make clean && make modules 2>&1", TEST_MODULE_DIR);
    
    ret = run_command(cmd, output, sizeof(output));
    if (ret != 0) {
        result->failed++;
        snprintf(result->error_msg, sizeof(result->error_msg),
                "Module compilation failed: %s", output);
        printf("  FAILED: %s\n", result->error_msg);
        return;
    }
    
    /* Check if module file was created */
    snprintf(cmd, sizeof(cmd), "ls %s/%s.ko", TEST_MODULE_DIR, TEST_MODULE_NAME);
    ret = run_command(cmd, NULL, 0);
    if (ret != 0) {
        result->failed++;
        snprintf(result->error_msg, sizeof(result->error_msg),
                "Module file %s.ko not found", TEST_MODULE_NAME);
        printf("  FAILED: %s\n", result->error_msg);
        return;
    }
    
    result->passed++;
    printf("  PASSED: Module compiled successfully\n");
}

/**
 * test_module_symbols - Test if required symbols are present
 * @result: Test result structure
 */
static void test_module_symbols(struct test_result *result)
{
    char cmd[512];
    char output[4096];
    int ret;
    const char *required_symbols[] = {
        "archangel_net_ai",
        "archangel_network_ai_init",
        "archangel_network_ai_cleanup",
        "archangel_ai_netfilter_hook_ipv4",
        "archangel_ai_classify_packet",
        "archangel_ai_extract_features",
        NULL
    };
    int i;
    
    printf("Testing network AI module symbols...\n");
    
    /* Use nm to check symbols in the module */
    snprintf(cmd, sizeof(cmd), "nm %s/%s.ko 2>/dev/null", TEST_MODULE_DIR, TEST_MODULE_NAME);
    
    ret = run_command(cmd, output, sizeof(output));
    if (ret != 0) {
        result->failed++;
        snprintf(result->error_msg, sizeof(result->error_msg),
                "Failed to read module symbols");
        printf("  FAILED: %s\n", result->error_msg);
        return;
    }
    
    /* Check for required symbols */
    for (i = 0; required_symbols[i]; i++) {
        if (!strstr(output, required_symbols[i])) {
            result->failed++;
            snprintf(result->error_msg, sizeof(result->error_msg),
                    "Required symbol '%s' not found", required_symbols[i]);
            printf("  FAILED: %s\n", result->error_msg);
            return;
        }
    }
    
    result->passed++;
    printf("  PASSED: All required symbols found\n");
}

/**
 * test_module_dependencies - Test module dependencies
 * @result: Test result structure
 */
static void test_module_dependencies(struct test_result *result)
{
    char cmd[512];
    char output[4096];
    int ret;
    
    printf("Testing network AI module dependencies...\n");
    
    /* Check module dependencies */
    snprintf(cmd, sizeof(cmd), "modinfo %s/%s.ko 2>/dev/null", TEST_MODULE_DIR, TEST_MODULE_NAME);
    
    ret = run_command(cmd, output, sizeof(output));
    if (ret != 0) {
        result->failed++;
        snprintf(result->error_msg, sizeof(result->error_msg),
                "Failed to read module info");
        printf("  FAILED: %s\n", result->error_msg);
        return;
    }
    
    /* Check for basic module information */
    if (!strstr(output, "filename:") || !strstr(output, "description:")) {
        result->failed++;
        snprintf(result->error_msg, sizeof(result->error_msg),
                "Module info incomplete");
        printf("  FAILED: %s\n", result->error_msg);
        return;
    }
    
    result->passed++;
    printf("  PASSED: Module dependencies check completed\n");
}

/**
 * test_header_includes - Test if header file is properly structured
 * @result: Test result structure
 */
static void test_header_includes(struct test_result *result)
{
    char cmd[512];
    char output[4096];
    int ret;
    const char *required_includes[] = {
        "#include <linux/module.h>",
        "#include <linux/netfilter.h>",
        "#include \"archangel_core.h\"",
        NULL
    };
    int i;
    
    printf("Testing network AI header file structure...\n");
    
    /* Read header file */
    snprintf(cmd, sizeof(cmd), "cat %s/archangel_network_ai.h", TEST_MODULE_DIR);
    
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
    if (!strstr(output, "#ifndef _ARCHANGEL_NETWORK_AI_H") ||
        !strstr(output, "#define _ARCHANGEL_NETWORK_AI_H") ||
        !strstr(output, "#endif")) {
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
        "struct archangel_network_ai_engine",
        "struct archangel_ml_classifier",
        "struct archangel_anomaly_detector",
        "struct archangel_stealth_engine",
        "struct archangel_packet_features",
        NULL
    };
    int i;
    
    printf("Testing network AI structure definitions...\n");
    
    /* Read header file */
    snprintf(cmd, sizeof(cmd), "cat %s/archangel_network_ai.h", TEST_MODULE_DIR);
    
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
        "enum archangel_net_decision",
        "enum archangel_packet_class",
        "enum archangel_stealth_mode",
        NULL
    };
    int i;
    
    printf("Testing network AI enum definitions...\n");
    
    /* Read header file */
    snprintf(cmd, sizeof(cmd), "cat %s/archangel_network_ai.h", TEST_MODULE_DIR);
    
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
 * cleanup_test_environment - Clean up after tests
 */
static void cleanup_test_environment(void)
{
    char cmd[512];
    
    printf("Cleaning up test environment...\n");
    
    /* Clean build artifacts */
    snprintf(cmd, sizeof(cmd), "cd %s && make clean >/dev/null 2>&1", TEST_MODULE_DIR);
    system(cmd);
    
    printf("Cleanup completed.\n");
}

/**
 * main - Main test function
 */
int main(void)
{
    struct test_result result = {0, 0, ""};
    
    printf("=== Archangel Network AI Module Compilation Test ===\n\n");
    
    /* Run all tests */
    test_header_includes(&result);
    test_structure_definitions(&result);
    test_enum_definitions(&result);
    test_module_compilation(&result);
    test_module_symbols(&result);
    test_module_dependencies(&result);
    
    /* Clean up */
    cleanup_test_environment();
    
    /* Print results */
    printf("\n=== Test Results ===\n");
    printf("Passed: %d\n", result.passed);
    printf("Failed: %d\n", result.failed);
    
    if (result.failed > 0) {
        printf("Last error: %s\n", result.error_msg);
        printf("\nNetwork AI module compilation test FAILED\n");
        return 1;
    }
    
    printf("\nNetwork AI module compilation test PASSED\n");
    return 0;
}