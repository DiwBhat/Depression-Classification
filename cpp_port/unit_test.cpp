#include <iostream>
#include <vector>
#include <string>

#include <sstream>   // Needed for formatting vector output
#include <algorithm> // For std::transform, std::sort, std::min_element, std::max_element, std::accumulate, std::find, std::distance, std::unique
#include <random>    // For shuffling, random feature selection, bootstrapping
#include <iomanip>   // For std::fixed, std::setprecision
#include <map>       // For counting class occurrences, user input map
#include <cmath>     // For std::sqrt, std::log2, std::fabs
#include <limits>    // For std::numeric_limits
#include <numeric>   // For std::iota


// Data Representation
using Sample = std::vector<double>;
using DatasetFeatures = std::vector<Sample>;
using DatasetLabels = std::vector<int>;

// Decision Tree Node Structure
struct TreeNode {
    bool is_leaf = false; int predicted_class = -1;
    int split_feature_index = -1; double split_value = 0.0;
    TreeNode* left_child = nullptr; TreeNode* right_child = nullptr;
    
    TreeNode(int cls = -1) : is_leaf(true), predicted_class(cls) {} 
    TreeNode(int feature_idx, double val) : split_feature_index(feature_idx), split_value(val) {}
    
    ~TreeNode() { 
        delete left_child; 
        delete right_child; 
    }
};

// Decision Tree Parameters
struct DecisionTreeParams {
    int max_depth = 10; 
    int min_samples_leaf = 5; 
    int num_features_to_consider = 0;
};

// Helper Functions 
double calculate_gini_impurity(const DatasetLabels& labels, const std::vector<size_t>& indices) {
    if (indices.empty()) return 0.0;
    std::map<int, int> class_counts;
    for (size_t idx : indices) {
        class_counts[labels[idx]]++;
    }
    double impurity = 1.0;
    for (auto const& pair : class_counts) {
        double prob = static_cast<double>(pair.second) / indices.size();
        impurity -= prob * prob;
    }
    return impurity;
}

int majority_class(const DatasetLabels& labels, const std::vector<size_t>& indices) {
    if (indices.empty()) return -1;
    std::map<int, int> class_counts;
    for (size_t idx : indices) {
        class_counts[labels[idx]]++;
    }
    int majority_cls = -1;
    int max_count = -1;
    for (auto const& pair : class_counts) {
        if (pair.second > max_count) {
            max_count = pair.second;
            majority_cls = pair.first;
        }
    }
    return majority_cls;
}

struct SplitInfo {
    int feature_index = -1;
    double split_value = 0.0;
    double gini_gain = -1.0; 
    std::vector<size_t> left_indices;
    std::vector<size_t> right_indices;
};

class DecisionTree {
public:
    TreeNode* root = nullptr;
    DecisionTreeParams params;
    std::mt19937 rng_dt; 

    DecisionTree(const DecisionTreeParams& p = DecisionTreeParams{}, unsigned int seed = 0) : params(p), rng_dt(seed) {}

    ~DecisionTree() {
        delete root; 
    }

    SplitInfo find_best_split(const DatasetFeatures& f, const DatasetLabels& l, 
                              const std::vector<size_t>& current_indices, 
                              const std::vector<int>& feature_subset_indices) {
        SplitInfo best_split_info; 
        best_split_info.gini_gain = -1.0; 

        if (current_indices.empty() || feature_subset_indices.empty() || f.empty()) {
            return best_split_info; 
        }
        
        double current_node_gini = calculate_gini_impurity(l, current_indices);

        if (current_indices.size() < 2 * static_cast<size_t>(params.min_samples_leaf) || current_indices.size() < 2) {
             return best_split_info;
        }

        for (int feature_idx : feature_subset_indices) {
            if (feature_idx < 0 || (f.empty() || feature_idx >= static_cast<int>(f[0].size()))) continue; 

            std::vector<double> unique_values;
            for (size_t sample_idx : current_indices) {
                unique_values.push_back(f[sample_idx][feature_idx]);
            }
            std::sort(unique_values.begin(), unique_values.end());
            unique_values.erase(std::unique(unique_values.begin(), unique_values.end()), unique_values.end());

            if (unique_values.size() < 2) { 
                continue;
            }

            for (size_t i = 0; i < unique_values.size() - 1; ++i) {
                double potential_split_value = (unique_values[i] + unique_values[i + 1]) / 2.0;
                std::vector<size_t> left_child_indices, right_child_indices;

                for (size_t sample_idx : current_indices) {
                    if (f[sample_idx][feature_idx] <= potential_split_value) {
                        left_child_indices.push_back(sample_idx);
                    } else {
                        right_child_indices.push_back(sample_idx);
                    }
                }

                if (left_child_indices.size() < static_cast<size_t>(params.min_samples_leaf) ||
                    right_child_indices.size() < static_cast<size_t>(params.min_samples_leaf)) {
                    continue; 
                }

                double gini_left = calculate_gini_impurity(l, left_child_indices);
                double gini_right = calculate_gini_impurity(l, right_child_indices);

                double p_left = static_cast<double>(left_child_indices.size()) / current_indices.size();
                double p_right = static_cast<double>(right_child_indices.size()) / current_indices.size();
                
                double current_split_gain = current_node_gini - (p_left * gini_left + p_right * gini_right);

                if (current_split_gain > best_split_info.gini_gain) {
                    best_split_info.gini_gain = current_split_gain;
                    best_split_info.feature_index = feature_idx;
                    best_split_info.split_value = potential_split_value;
                    best_split_info.left_indices = left_child_indices;
                    best_split_info.right_indices = right_child_indices;
                }
            }
        }
        return best_split_info;
    }
};


// Unit Test Framework 
const double EPSILON = 1e-9;

// Helper to print vectors (for error messages)
template <typename T>
std::string vector_to_string(const std::vector<T>& vec) {
    std::stringstream ss;
    ss << "{";
    for (size_t i = 0; i < vec.size(); ++i) {
        ss << vec[i];
        if (i < vec.size() - 1) {
            ss << ", ";
        }
    }
    ss << "}";
    return ss.str();
}


bool check_near(double val1, double val2, const std::string& test_name, const std::string& var_name, double epsilon = EPSILON) {
    if (std::fabs(val1 - val2) >= epsilon) {
        std::cerr << "\n  [" << test_name << "] FAIL: " << var_name 
                  << ". Expected near " << val2 << ", Got: " << val1 << std::endl;
        return false;
    }
    return true;
}

// Overload check_equal for basic types
template <typename T>
typename std::enable_if<!std::is_same<T, std::vector<size_t>>::value, bool>::type // SFINAE to exclude vector
check_equal(T val1, T val2, const std::string& test_name, const std::string& var_name) {
    if (val1 != val2) {
        std::cerr << "\n  [" << test_name << "] FAIL: " << var_name 
                  << ". Expected " << val2 << ", Got: " << val1 << std::endl;
        return false;
    }
    return true;
}

// Specific overload for std::vector<size_t>
bool check_equal(const std::vector<size_t>& val1, const std::vector<size_t>& val2, const std::string& test_name, const std::string& var_name) {
    if (val1 != val2) {
        std::cerr << "\n  [" << test_name << "] FAIL: " << var_name 
                  << ". Expected " << vector_to_string(val2) << ", Got: " << vector_to_string(val1) << std::endl;
        return false;
    }
    return true;
}


template <typename T>
bool check_gt(T val1, T val2, const std::string& test_name, const std::string& var_name) {
    if (!(val1 > val2)) {
        std::cerr << "\n  [" << test_name << "] FAIL: " << var_name 
                  << ". Expected > " << val2 << ", Got: " << val1 << std::endl;
        return false;
    }
    return true;
}


// Test Runner State
int tests_run = 0;
int tests_passed = 0;

// Macro for running a test and reporting
#define RUN_TEST(test_function, test_name_str) \
    tests_run++; \
    std::cout << "Running " << test_name_str << "... "; \
    if (test_function(test_name_str)) { \
        tests_passed++; \
        std::cout << "PASS" << std::endl; \
    } else { \
        std::cout << "FAIL" << std::endl; \
    }

// Test Functions 

// Test Case ID: UT_001
bool test_GiniImpurity_BalancedTwoClass(const std::string& test_name) {
    DatasetLabels labels = {0, 0, 1, 1};
    std::vector<size_t> indices = {0, 1, 2, 3};
    double expected_gini = 0.5;
    double actual_gini = calculate_gini_impurity(labels, indices);
    return check_near(actual_gini, expected_gini, test_name, "Gini Impurity");
}

// Test Case ID: UT_002
bool test_GiniImpurity_PureNodeClass0(const std::string& test_name) {
    DatasetLabels labels = {0, 0, 0, 0};
    std::vector<size_t> indices = {0, 1, 2, 3};
    double expected_gini = 0.0;
    double actual_gini = calculate_gini_impurity(labels, indices);
    return check_near(actual_gini, expected_gini, test_name, "Gini Impurity");
}

// Test Case ID: UT_003
bool test_MajorityClass_MixedSet(const std::string& test_name) {
    DatasetLabels labels = {0, 1, 1, 0, 1};
    std::vector<size_t> indices = {0, 1, 2, 3, 4};
    int expected_majority = 1;
    int actual_majority = majority_class(labels, indices);
    return check_equal(actual_majority, expected_majority, test_name, "Majority Class");
}

// Test Case ID: UT_004
bool test_FindBestSplit_SimpleDataset(const std::string& test_name) {
    DecisionTreeParams params_for_split_test;
    params_for_split_test.min_samples_leaf = 1; 
    params_for_split_test.num_features_to_consider = 2; 
    DecisionTree tree(params_for_split_test);

    DatasetFeatures features = {{1.0, 10.0}, {2.0, 20.0}, {3.0, 10.0}, {4.0, 20.0}};
    DatasetLabels labels = {0, 1, 0, 1};
    std::vector<size_t> current_indices = {0, 1, 2, 3};
    std::vector<int> feature_subset_indices = {0, 1}; 

    SplitInfo best_split = tree.find_best_split(features, labels, current_indices, feature_subset_indices);

    bool overall_pass = true;
    overall_pass &= check_gt(best_split.gini_gain, 0.0, test_name, "Gini Gain");
    overall_pass &= check_equal(best_split.feature_index, 1, test_name, "Feature Index");
    overall_pass &= check_near(best_split.split_value, 15.0, test_name, "Split Value");
    overall_pass &= check_equal(best_split.left_indices.size(), (size_t)2, test_name, "Left Indices Size");
    overall_pass &= check_equal(best_split.right_indices.size(), (size_t)2, test_name, "Right Indices Size");

    std::vector<size_t> expected_left = {0, 2}; 
    std::vector<size_t> expected_right = {1, 3};
    std::vector<size_t> actual_left = best_split.left_indices; 
    std::vector<size_t> actual_right = best_split.right_indices;
    std::sort(actual_left.begin(), actual_left.end());
    std::sort(actual_right.begin(), actual_right.end());
    
    overall_pass &= check_equal(actual_left, expected_left, test_name, "Left Indices Content");
    overall_pass &= check_equal(actual_right, expected_right, test_name, "Right Indices Content");
    
    return overall_pass;
}


int main() {
    std::cout << "--- Running Unit Tests ---" << std::endl;

    RUN_TEST(test_GiniImpurity_BalancedTwoClass, "UT_001_Gini_Balanced");
    RUN_TEST(test_GiniImpurity_PureNodeClass0, "UT_002_Gini_Pure");
    RUN_TEST(test_MajorityClass_MixedSet, "UT_003_Majority_Mixed");
    RUN_TEST(test_FindBestSplit_SimpleDataset, "UT_004_FindBestSplit_Simple");

    std::cout << "\n--- Test Summary ---" << std::endl;
    std::cout << "Total Tests Run: " << tests_run << std::endl;
    std::cout << "Tests Passed:    " << tests_passed << std::endl;
    std::cout << "Tests Failed:    " << (tests_run - tests_passed) << std::endl;

    if (tests_run - tests_passed != 0) {
        std::cout << "\nSOME TESTS FAILED!" << std::endl;
    } else {
        std::cout << "\nALL TESTS PASSED!" << std::endl;
    }

    return (tests_run - tests_passed == 0) ? 0 : 1;
}