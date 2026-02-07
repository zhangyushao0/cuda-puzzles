#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <random>

// Tolerance-based comparison for floats
// Formula: |a - b| <= atol + rtol * max(|a|, |b|)
inline bool check_close(float a, float b, float atol = 1e-4f, float rtol = 1e-4f) {
  float diff = fabs(a - b);
  float tolerance = atol + rtol * fmax(fabs(a), fabs(b));
  return diff <= tolerance;
}

// Array comparison with detailed diff on failure
// Returns true only if ALL elements pass tolerance check
inline bool check_array_close(const float* got, const float* expected, int n, 
                               float atol = 1e-4f, float rtol = 1e-4f,
                               bool verbose = false) {
  bool all_pass = true;
  int fail_count = 0;
  const int max_failures_to_print = 10;
  
  for (int i = 0; i < n; i++) {
    if (!check_close(got[i], expected[i], atol, rtol)) {
      if (fail_count < max_failures_to_print || verbose) {
        float diff = fabs(got[i] - expected[i]);
        float tolerance = atol + rtol * fmax(fabs(expected[i]), fabs(got[i]));
        fprintf(stderr, "  [FAIL] Index %d: Expected %.6f, Got %.6f, Diff %.6f (tolerance %.6f)\n",
                i, expected[i], got[i], diff, tolerance);
      }
      fail_count++;
      all_pass = false;
    }
  }
  
  if (fail_count > max_failures_to_print && !verbose) {
    fprintf(stderr, "  ... and %d more failures (total %d/%d failed)\n", 
            fail_count - max_failures_to_print, fail_count, n);
  }
  
  return all_pass;
}

// Test result output with formatted output
inline void print_test_result(const char* name, bool passed, const char* details = nullptr) {
  if (passed) {
    printf("[PASS] %s\n", name);
  } else {
    printf("[FAIL] %s", name);
    if (details) {
      printf(": %s", details);
    }
    printf("\n");
  }
}

// Lightweight test framework
struct TestCase {
  const char* name;
  void (*func)();
  bool passed;
};

static std::vector<TestCase>& get_test_registry() {
  static std::vector<TestCase> tests;
  return tests;
}

// Test case registration macro
#define TEST_CASE(test_name) \
  static void test_##test_name(); \
  static struct TestRegistrar_##test_name { \
    TestRegistrar_##test_name() { \
      TestCase tc; \
      tc.name = #test_name; \
      tc.func = test_##test_name; \
      tc.passed = false; \
      get_test_registry().push_back(tc); \
    } \
  } test_registrar_##test_name; \
  static void test_##test_name()

// Run all registered tests
inline int RUN_ALL_TESTS() {
  auto& tests = get_test_registry();
  int passed = 0;
  int total = tests.size();
  
  printf("\n=== Running %d tests ===\n\n", total);
  
  for (auto& test : tests) {
    try {
      test.func();
      test.passed = true;
      printf("[PASS] %s\n", test.name);
      passed++;
    } catch (const std::exception& e) {
      test.passed = false;
      printf("[FAIL] %s: %s\n", test.name, e.what());
    } catch (...) {
      test.passed = false;
      printf("[FAIL] %s: Unknown exception\n", test.name);
    }
  }
  
  printf("\n=== %d/%d tests passed ===\n", passed, total);
  
  return (passed == total) ? 0 : 1;
}

// Deterministic random seed setup
inline void set_seed(unsigned int seed) {
  srand(seed);
}

// Deterministic host-side random fill
inline void fill_random(float* arr, int n, unsigned int seed, float min = 0.0f, float max = 1.0f) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dis(min, max);
  for (int i = 0; i < n; i++) {
    arr[i] = dis(gen);
  }
}

#endif // TEST_UTILS_H
