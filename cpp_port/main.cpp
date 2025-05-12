#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm> // For std::transform, std::sort, std::min_element, std::max_element, std::accumulate, std::find, std::distance
#include <random>    // For shuffling, random feature selection, bootstrapping
#include <iomanip>   // For std::fixed, std::setprecision
#include <map>       // For counting class occurrences, user input map
#include <cmath>     // For std::sqrt, std::log2
#include <limits>    // For std::numeric_limits
#include <numeric>   // For std::iota
#include <chrono>  // For date and time
#include <ctime>   // For formatting time

// Configuration
const int NUM_CLASSES_CONST = 2; 

// Utility Functions
std::string trim(const std::string& str) {
    const std::string whitespace = " \t\n\r\f\v";
    size_t start = str.find_first_not_of(whitespace);
    if (std::string::npos == start) return "";
    size_t end = str.find_last_not_of(whitespace);
    return str.substr(start, end - start + 1);
}

// Data Representation
using Sample = std::vector<double>;
using DatasetFeatures = std::vector<Sample>;
using DatasetLabels = std::vector<int>;

// Decision Tree Node Structure
struct TreeNode {
    bool is_leaf = false; int predicted_class = -1;
    int split_feature_index = -1; double split_value = 0.0;
    TreeNode* left_child = nullptr; TreeNode* right_child = nullptr;
    TreeNode(int cls) : is_leaf(true), predicted_class(cls) {}
    TreeNode(int feature_idx, double val) : split_feature_index(feature_idx), split_value(val) {}
    ~TreeNode() { delete left_child; delete right_child; }
};

// Decision Tree Parameters
struct DecisionTreeParams {
    int max_depth = 10; int min_samples_leaf = 5; int num_features_to_consider = 0;
};

// Decision Tree Helper Functions
double calculate_gini_impurity(const DatasetLabels& labels, const std::vector<size_t>& indices) {
    if (indices.empty()) return 0.0; std::map<int, int> class_counts;
    for (size_t idx : indices) class_counts[labels[idx]]++; double impurity = 1.0;
    for (auto const& [label, count] : class_counts) {
        double prob = static_cast<double>(count) / indices.size(); impurity -= prob * prob;
    } return impurity;
}
int majority_class(const DatasetLabels& labels, const std::vector<size_t>& indices) {
    if (indices.empty()) return -1; std::map<int, int> class_counts;
    for (size_t idx : indices) class_counts[labels[idx]]++; int majority_cls = -1; int max_count = -1;
    for (auto const& [label, count] : class_counts) {
        if (count > max_count) { max_count = count; majority_cls = label; }
    } return majority_cls;
}
struct SplitInfo {
    int feature_index = -1; double split_value = 0.0; double gini_gain = -1.0;
    std::vector<size_t> left_indices; std::vector<size_t> right_indices;
};

// Decision Tree Class
class DecisionTree {
public:
    TreeNode* root = nullptr; DecisionTreeParams params; std::mt19937 rng_dt;
    DecisionTree(const DecisionTreeParams& p, unsigned int seed) : params(p), rng_dt(seed) {}
    ~DecisionTree() { delete root; }
    SplitInfo find_best_split(const DatasetFeatures& f, const DatasetLabels& l, const std::vector<size_t>& ci, const std::vector<int>& fsi) {
        SplitInfo bs; bs.gini_gain = -1.0; double cng = calculate_gini_impurity(l, ci);
        if (ci.size() < 2 * static_cast<size_t>(params.min_samples_leaf)) return bs;
        for (int fi : fsi) {
            std::vector<double> uv; for (size_t si : ci) uv.push_back(f[si][fi]);
            std::sort(uv.begin(), uv.end()); uv.erase(std::unique(uv.begin(), uv.end()), uv.end());
            for (size_t i = 0; i < uv.size() - 1; ++i) {
                double psv = (uv[i] + uv[i+1]) / 2.0; std::vector<size_t> lci, rci;
                for (size_t si : ci) { if (f[si][fi] <= psv) lci.push_back(si); else rci.push_back(si); }
                if (lci.size() < static_cast<size_t>(params.min_samples_leaf) || rci.size() < static_cast<size_t>(params.min_samples_leaf)) continue;
                double gl = calculate_gini_impurity(l, lci); double gr = calculate_gini_impurity(l, rci);
                double pl = static_cast<double>(lci.size())/ci.size(); double pr = static_cast<double>(rci.size())/ci.size();
                double csg = cng - (pl*gl + pr*gr);
                if (csg > bs.gini_gain) { bs.gini_gain=csg; bs.feature_index=fi; bs.split_value=psv; bs.left_indices=lci; bs.right_indices=rci; }
            }
        } return bs;
    }
    TreeNode* build_tree_recursive(const DatasetFeatures& f, const DatasetLabels& l, const std::vector<size_t>& ci, int cd) {
        double cg = calculate_gini_impurity(l, ci);
        if (cg == 0.0 || cd >= params.max_depth || ci.size() < static_cast<size_t>(params.min_samples_leaf)*2 || ci.size() < 2)
            return new TreeNode(majority_class(l, ci));
        std::vector<int> afi(f[0].size()); std::iota(afi.begin(), afi.end(), 0); std::shuffle(afi.begin(), afi.end(), rng_dt);
        int nfu = params.num_features_to_consider;
        if (nfu <= 0 || nfu > static_cast<int>(f[0].size())) {
            nfu = static_cast<int>(std::sqrt(f[0].size())); if (nfu == 0 && !f[0].empty()) nfu=1; else if (f[0].empty()) nfu=0;
        } nfu = std::min(nfu, static_cast<int>(afi.size()));
        std::vector<int> fs; if (nfu > 0) fs.assign(afi.begin(), afi.begin() + nfu); else if (!afi.empty()) fs.push_back(afi[0]);
        if (fs.empty()) return new TreeNode(majority_class(l, ci));
        SplitInfo bsi = find_best_split(f, l, ci, fs);
        if (bsi.gini_gain <= 0.0) return new TreeNode(majority_class(l, ci));
        TreeNode* n = new TreeNode(bsi.feature_index, bsi.split_value);
        n->left_child = build_tree_recursive(f, l, bsi.left_indices, cd + 1);
        n->right_child = build_tree_recursive(f, l, bsi.right_indices, cd + 1);
        return n;
    }
    void train(const DatasetFeatures& f, const DatasetLabels& l) {
        if (f.empty() || l.empty()) return; std::vector<size_t> ii(f.size()); std::iota(ii.begin(),ii.end(),0);
        delete root; root = nullptr; root = build_tree_recursive(f, l, ii, 0);
    }
    int predict_sample(const Sample& s, TreeNode* n) const {
        if (!n) return -1; if (n->is_leaf) return n->predicted_class;
        if (s[n->split_feature_index] <= n->split_value) return predict_sample(s, n->left_child);
        else return predict_sample(s, n->right_child);
    }
    int predict(const Sample& s) const { return predict_sample(s, root); }
};

// Random Forest Parameters
struct RandomForestParams {
    int num_trees = 100; DecisionTreeParams tree_params; double bootstrap_sample_ratio = 1.0; unsigned int random_seed = 42;
};

// Random Forest Class
class RandomForest {
public:
    std::vector<DecisionTree*> trees; RandomForestParams params; std::vector<std::string> feature_names_internal;
    RandomForest(const RandomForestParams& p) : params(p) {}
    ~RandomForest() { for (DecisionTree* t : trees) delete t; trees.clear(); }
    std::vector<size_t> create_bootstrap_sample(size_t os, std::mt19937& g) {
        std::vector<size_t> bi; size_t ss = static_cast<size_t>(os*params.bootstrap_sample_ratio); if(ss==0 && os>0) ss=1;
        std::uniform_int_distribution<size_t> d(0, os -1); for(size_t i=0;i<ss;++i) bi.push_back(d(g)); return bi;
    }
    void train(const DatasetFeatures& f, const DatasetLabels& l, const std::vector<std::string>& fn) {
        if (f.empty()) return; feature_names_internal = fn; for (DecisionTree* t : trees) delete t; trees.clear();
        std::mt19937 rng_rf(params.random_seed);
        if (params.tree_params.num_features_to_consider <= 0 && !f.empty() && !f[0].empty()) {
            params.tree_params.num_features_to_consider = static_cast<int>(std::sqrt(f[0].size()));
            if (params.tree_params.num_features_to_consider == 0) params.tree_params.num_features_to_consider = 1;
        }
        std::cout << "Training Random Forest with " << params.num_trees << " trees." << std::endl;
        for (int i = 0; i < params.num_trees; ++i) {
            std::vector<size_t> bsi = create_bootstrap_sample(f.size(), rng_rf);
            DatasetFeatures tfs; DatasetLabels tls; for(size_t idx:bsi){tfs.push_back(f[idx]);tls.push_back(l[idx]);}
            if (tfs.empty()) {std::cout<<"Skip tree "<<i+1<<" empty bootstrap."<<std::endl; continue;}
            DecisionTree* t = new DecisionTree(params.tree_params, params.random_seed + i); t->train(tfs, tls); trees.push_back(t);
            if ((i+1)%10==0||i==params.num_trees-1) std::cout<<"Trained tree "<<(i+1)<<"/"<<params.num_trees<<std::endl;
        } std::cout << "Random Forest training complete." << std::endl;
    }
    int predict_class(const Sample& s) const {
        if (trees.empty()) return -1; std::map<int, int> v; for(const DecisionTree* t:trees)v[t->predict(s)]++;
        int mc=-1; int mv=-1; for(auto const& [c,cnt]:v){if(cnt>mv){mv=cnt;mc=c;}} return mc;
    }
    double predict_probability_class1(const Sample& s) const {
        if (trees.empty()) return 0.0; int v1=0; for(const DecisionTree* t:trees){if(t->predict(s)==1)v1++;}
        return static_cast<double>(v1)/trees.size();
    }
};

// Data Loading
bool load_numeric_csv(const std::string& filename, DatasetFeatures& features, DatasetLabels& labels, std::vector<std::string>& feature_names_output) {
    std::ifstream infile(filename);
    if (!infile) { std::cerr << "Error opening data file: " << filename << std::endl; return false; }
    std::string line; std::vector<std::string> all_header_names;
    if (std::getline(infile, line)) {
        std::stringstream ss_header(line); std::string col_name;
        while (std::getline(ss_header, col_name, ',')) all_header_names.push_back(trim(col_name));
        if (all_header_names.empty()) { std::cerr << "Error: Empty header." << std::endl; return false; }
        feature_names_output.assign(all_header_names.begin(), all_header_names.end() - 1);
    } else { std::cerr << "Error: Could not read header." << std::endl; return false; }

    while (std::getline(infile, line)) {
        std::stringstream ss_row(line); std::string cell_str;
        Sample current_features; int label = 0; size_t col_idx = 0;
        while (std::getline(ss_row, cell_str, ',')) {
            std::string t_cell = trim(cell_str); double val;
            try {
                if (t_cell == "TRUE" || t_cell == "True" || t_cell == "true") val = 1.0;
                else if (t_cell == "FALSE" || t_cell == "False" || t_cell == "false") val = 0.0;
                else val = std::stod(t_cell);

                if (col_idx < feature_names_output.size()) current_features.push_back(val);
                else if (col_idx == feature_names_output.size()) label = static_cast<int>(val);
                else { throw std::runtime_error("Too many columns in row"); }
            } catch (const std::exception& e) {
                std::cerr << "Warn: Invalid data '" << t_cell << "' in row. Skipping. Err: " << e.what() << std::endl;
                current_features.clear(); goto next_row_main_load_final;
            }
            col_idx++;
        }
        if (col_idx == all_header_names.size() && !current_features.empty()) {
            features.push_back(current_features); labels.push_back(label);
        } else if (col_idx != 0) {
            std::cerr << "Warn: Row col count mismatch (" << col_idx << " vs " << all_header_names.size() << "). Skipping." << std::endl;
        }
        next_row_main_load_final:;
    }
    infile.close(); return !features.empty();
}

// Data Splitting
void split_data(const DatasetFeatures& af, const DatasetLabels& al, DatasetFeatures& trf, DatasetLabels& trl, DatasetFeatures& tef, DatasetLabels& tel, double tr_r = 0.8, unsigned int rs = 42) { 
    if (af.size()!=al.size()||af.empty()) {std::cerr<<"Err: Invalid data for split.\n"; return;}
    std::vector<size_t> idx(af.size()); std::iota(idx.begin(),idx.end(),0);
    std::mt19937 g(rs); std::shuffle(idx.begin(),idx.end(),g);
    size_t tr_s = static_cast<size_t>(af.size()*tr_r);
    trf.clear();trl.clear();tef.clear();tel.clear();
    for(size_t i=0;i<af.size();++i){
        if(i<tr_s){trf.push_back(af[idx[i]]);trl.push_back(al[idx[i]]);}
        else{tef.push_back(af[idx[i]]);tel.push_back(al[idx[i]]);}
    }
}

// CLI Input and Feature Transformation 
const std::vector<std::pair<int, std::string>> cli_sleep_options = {{0,"Less than 5 hours"},{1,"5-6 hours"},{2,"7-8 hours"},{3,"More than 8 hours"}};
const std::vector<std::pair<int, std::string>> cli_degree_type_options = {{0,"Science/Tech Related Field"},{1,"Non-Science/Arts/Business/Law Field"}};
const std::vector<double> cli_age_bins = {17.0, 24.0, 30.0, 40.0}; 
const std::vector<std::string> cli_age_labels = {"18-24", "25-30", "31-39"};
const std::string cli_dropped_ohe_age_group_col = "Age_Group_18-24";
const std::string cli_dropped_ohe_gender_col = "Gender_Female"; 
const std::string cli_dropped_ohe_degree_type_col = "Degree_Type_Non-Science/Arts/Business/Law";

// History File Configuration
const std::string HISTORY_FILE_NAME = "risk_history.txt";
const int MAX_HISTORY_ENTRIES_TO_DISPLAY = 5; // Show last 5 entries

// History Management Functions

// Structure to hold a history entry
struct HistoryEntry {
    std::string timestamp;
    double probability_class1;
    std::string risk_level_str; // "HIGH" or "LOW"
};

// Function to get current timestamp as string
std::string get_current_timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm buf;
#ifdef _WIN32
    localtime_s(&buf, &now_time); // Windows specific
#else
    localtime_r(&now_time, &buf); // POSIX specific
#endif
    std::stringstream ss;
    ss << std::put_time(&buf, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

// Function to read risk history
std::vector<HistoryEntry> read_risk_history() {
    std::vector<HistoryEntry> history;
    std::ifstream infile(HISTORY_FILE_NAME);
    std::string line;
    if (infile.is_open()) {
        while (std::getline(infile, line)) {
            std::stringstream ss(line);
            std::string segment;
            HistoryEntry entry;
            try {
                // Expected format: Timestamp,Probability,RiskLevelString
                if (std::getline(ss, segment, ',')) entry.timestamp = trim(segment); else continue;
                if (std::getline(ss, segment, ',')) entry.probability_class1 = std::stod(trim(segment)); else continue;
                if (std::getline(ss, segment, ',')) entry.risk_level_str = trim(segment); else continue;
                history.push_back(entry);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not parse history line: " << line << " | Error: " << e.what() << std::endl;
            }
        }
        infile.close();
    }
    return history;
}

// Function to append to risk history
void append_to_risk_history(const HistoryEntry& new_entry) {
    std::ofstream outfile(HISTORY_FILE_NAME, std::ios_base::app); 
    if (outfile.is_open()) {
        outfile << new_entry.timestamp << ","
                << std::fixed << std::setprecision(4) << new_entry.probability_class1 << ","
                << new_entry.risk_level_str << std::endl;
        outfile.close();
    } else {
        std::cerr << "Warning: Could not open history file to append data." << std::endl;
    }
}

// Function to display recent history
void display_recent_history(const std::vector<HistoryEntry>& history) {
    if (history.empty()) {
        std::cout << "No past prediction history found." << std::endl;
        return;
    }
    std::cout << "\n--- Recent Prediction History (Last " << MAX_HISTORY_ENTRIES_TO_DISPLAY << ") ---" << std::endl;
    size_t start_index = history.size() > MAX_HISTORY_ENTRIES_TO_DISPLAY ? history.size() - MAX_HISTORY_ENTRIES_TO_DISPLAY : 0;
    for (size_t i = start_index; i < history.size(); ++i) {
        std::cout << history[i].timestamp << " - Risk: " << history[i].risk_level_str
                  << " (Probability of High Risk: " << std::fixed << std::setprecision(3) 
                  << history[i].probability_class1 << ")" << std::endl;
    }
    std::cout << "--------------------------------------" << std::endl;
}


Sample get_simplified_user_input_and_transform(const std::vector<std::string>& all_model_feature_names) {
    std::map<std::string, double> raw_inputs; Sample final_feature_vector(all_model_feature_names.size(), 0.0);
    std::cout << "\nPlease enter student details (type 'quit' at any prompt):" << std::endl;
    auto get_double = [&](const std::string& p, double min_v, double max_v){/*...*/ return 0.0;}; 
    auto get_int_choice = [&](const std::string& p, const std::vector<std::pair<int,std::string>>& opts){/*...*/ return 0;}; 

    std::cout << "Age (17-70): "; std::cin >> raw_inputs["Age"];
    std::cout << "Academic Pressure (1-5): "; std::cin >> raw_inputs["Academic Pressure"];
    std::cout << "CGPA (0-10): "; std::cin >> raw_inputs["CGPA"];
    std::cout << "Study Satisfaction (1-5): "; std::cin >> raw_inputs["Study Satisfaction"];
    std::cout << "Suicidal Thoughts (0=No, 1=Yes): "; std::cin >> raw_inputs["Suicidal_Thoughts"];
    std::cout << "Work/Study Hours per day (0-24): "; std::cin >> raw_inputs["Work/Study Hours"];
    std::cout << "Financial Stress (1-5): "; std::cin >> raw_inputs["Financial Stress"];
    std::cout << "Family History of Mental Illness (0=No, 1=Yes): "; std::cin >> raw_inputs["Family History of Mental Illness"];
    std::cout << "Sleep Ordinal (0:<5h, 1:5-6h, 2:7-8h, 3:>8h): "; std::cin >> raw_inputs["Sleep_Ordinal"];
    std::cout << "Gender (0=Female/Other, 1=Male): "; std::cin >> raw_inputs["_gender_choice"]; // 0 or 1
    std::cout << "Degree Type (0=Sci/Tech, 1=Non-Sci/Arts/Biz/Law): "; std::cin >> raw_inputs["_degree_type_choice"];
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');


    for (size_t i = 0; i < all_model_feature_names.size(); ++i) {
        const std::string& fname = all_model_feature_names[i];
        if(fname == "Age") final_feature_vector[i] = raw_inputs["Age"];
        else if(fname == "Academic Pressure") final_feature_vector[i] = raw_inputs["Academic Pressure"];
        else if(fname == "CGPA") final_feature_vector[i] = raw_inputs["CGPA"];
        else if(fname == "Study Satisfaction") final_feature_vector[i] = raw_inputs["Study Satisfaction"];
        else if(fname == "Suicidal_Thoughts") final_feature_vector[i] = raw_inputs["Suicidal_Thoughts"];
        else if(fname == "Work/Study Hours") final_feature_vector[i] = raw_inputs["Work/Study Hours"];
        else if(fname == "Financial Stress") final_feature_vector[i] = raw_inputs["Financial Stress"];
        else if(fname == "Family History of Mental Illness") final_feature_vector[i] = raw_inputs["Family History of Mental Illness"];
        else if(fname == "Sleep_Ordinal") final_feature_vector[i] = raw_inputs["Sleep_Ordinal"];
        else if(fname == "Total_Stress") final_feature_vector[i] = raw_inputs["Academic Pressure"] + raw_inputs["Financial Stress"];
        else if (fname == "Gender_Male" && raw_inputs["_gender_choice"] == 1.0) final_feature_vector[i] = 1.0;
        // Handle Age_Group_*
        else if (fname.rfind("Age_Group_", 0) == 0) {
            double age_val = raw_inputs["Age"]; std::string cur_label = "";
            for (size_t j = 0; j < cli_age_bins.size() - 1; ++j) if (age_val > cli_age_bins[j] && age_val <= cli_age_bins[j+1]) { cur_label = cli_age_labels[j]; break; }
            if (cur_label.empty() && age_val > cli_age_bins.back()) cur_label = cli_age_labels.back();
            if (fname == ("Age_Group_" + cur_label) && fname != cli_dropped_ohe_age_group_col) final_feature_vector[i] = 1.0;
        }
        // Handle Degree_Type_*
        else if (fname == "Degree_Type_Science/Tech" && raw_inputs["_degree_type_choice"] == 0.0 && fname != cli_dropped_ohe_degree_type_col) final_feature_vector[i] = 1.0;
        else if (fname == "Degree_Type_Non-Science/Arts/Business/Law" && raw_inputs["_degree_type_choice"] == 1.0 && fname != cli_dropped_ohe_degree_type_col) final_feature_vector[i] = 1.0;
    }
    return final_feature_vector;
}

// Recommendation Function 
double get_feature_value_by_name(const Sample& sf, const std::vector<std::string>& afn, const std::string& tfn, double dv = 0.0) {
    for (size_t i=0;i<afn.size();++i){ if(afn[i]==tfn){ if(i<sf.size())return sf[i]; else return dv;}} return dv;
}
void display_recommendations(int pc, const Sample& uaf, const std::vector<std::string>& afn) {
    std::cout << "\n--- Recommendations ---" << std::endl;
    if (pc == 1) { std::cout << "- **It's important to reach out:** Consider speaking with a counselor or therapist." << std::endl; }
    else { std::cout << "- **Continue to prioritize your well-being!**" << std::endl;} std::cout<<std::endl;
    double ap = get_feature_value_by_name(uaf,afn,"Academic Pressure"); if(ap>3) std::cout<<"**Regarding Academic Pressure ("<<static_cast<int>(ap)<<"/5):**\n  - Break tasks down.\n  - Practice time management.\n";
    double cg = get_feature_value_by_name(uaf,afn,"CGPA"); if(cg<7.0) std::cout<<"**Regarding Study Habits (CGPA: "<<std::fixed<<std::setprecision(2)<<cg<<"):\n  - Find effective study methods.\n  - Consider study groups.\n";
    double wsh= get_feature_value_by_name(uaf,afn,"Work/Study Hours"); if(wsh>8) std::cout<<"**Regarding Work/Study Hours ("<<static_cast<int>(wsh)<<"h/day):\n  - Evaluate sustainability.\n  - Schedule rest.\n";
    double fs = get_feature_value_by_name(uaf,afn,"Financial Stress"); if(fs>3) std::cout<<"**Regarding Financial Stress ("<<static_cast<int>(fs)<<"/5):\n  - Create a budget.\n  - Explore financial aid resources.\n";
    if(get_feature_value_by_name(uaf,afn,"Suicidal_Thoughts")==1.0) std::cout<<"\n**IMPORTANT: Suicidal Thoughts:**\n  - **Please reach out immediately for help (e.g., crisis hotline 988 in USA, counseling services).**\n";
    if(get_feature_value_by_name(uaf,afn,"Sleep_Ordinal")<2.0) std::cout<<"**Regarding Sleep:**\n  - Aim for a consistent schedule.\n  - Create a relaxing bedtime routine.\n";
}


// Main Function
int main() {
    std::ios_base::sync_with_stdio(false); 
    std::cin.tie(NULL);

    std::cout << "Student Depression Risk Prediction (Random Forest from Scratch)\n";
    std::cout << "-------------------------------------------------------------\n";

    DatasetFeatures all_features; DatasetLabels all_labels; std::vector<std::string> feature_names_list;
    const std::string data_path = "cleaned_student_data.csv";

    if (!load_numeric_csv(data_path, all_features, all_labels, feature_names_list)) return 1;
    std::cout << "Data loaded: " << all_features.size() << " samples, "
              << (all_features.empty() ? 0 : all_features[0].size()) << " features." << std::endl;
    if (all_features.empty()) return 1;

    DatasetFeatures train_f, test_f; DatasetLabels train_l, test_l;
    split_data(all_features, all_labels, train_f, train_l, test_f, test_l, 0.8, 123);
    std::cout << "Training set: " << train_f.size() << ", Test set: " << test_f.size() << std::endl;
    if (train_f.empty() || test_f.empty()) { std::cerr << "Data split failed." << std::endl; return 1; }

    RandomForestParams rf_params; rf_params.num_trees = 50; rf_params.tree_params.max_depth = 8;
    rf_params.tree_params.min_samples_leaf = 5; rf_params.random_seed = 42;
    RandomForest model(rf_params);
    std::cout << "\nTraining Random Forest model..." << std::endl;
    model.train(train_f, train_l, feature_names_list);

    std::cout << "\n--- Evaluating Model on Test Set ---" << std::endl;
    if (!test_f.empty()) {
        int correct = 0; unsigned int tp=0, tn=0, fp=0, fn=0;
        for (size_t i = 0; i < test_f.size(); ++i) {
            int pred = model.predict_class(test_f[i]); if (pred == test_l[i]) correct++;
            if (test_l[i] == 1 && pred == 1) tp++; else if (test_l[i] == 0 && pred == 0) tn++;
            else if (test_l[i] == 0 && pred == 1) fp++; else if (test_l[i] == 1 && pred == 0) fn++;
        }
        double acc = static_cast<double>(correct) / test_f.size();
        std::cout << "Test Accuracy: " << std::fixed << std::setprecision(4) << acc * 100.0 << "%" << std::endl;
        std::cout << "\nConfusion Matrix (Test Set):\n          Predicted 0   Predicted 1\n";
        std::cout << "Actual 0      " << std::setw(5) << tn << "         " << std::setw(5) << fp << std::endl;
        std::cout << "Actual 1      " << std::setw(5) << fn << "         " << std::setw(5) << tp << std::endl;
    }

    std::cout << "\n\n--- Interactive Prediction (Single User History) ---" << std::endl;
    std::string choice;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); 

    while (true) {
        std::cout << "\nMake a new prediction? (yes/no or quit): "; 
        std::getline(std::cin, choice);
        std::transform(choice.begin(), choice.end(), choice.begin(), ::tolower);

        if (choice == "no" || choice == "quit") {
            break;
        }
        if (choice != "yes") {
            std::cout << "Invalid choice. Please type 'yes', 'no', or 'quit'." << std::endl;
            continue;
        }

        // Display history before asking for new input
        std::vector<HistoryEntry> past_history = read_risk_history();
        display_recent_history(past_history);

        try {
            Sample user_f = get_simplified_user_input_and_transform(feature_names_list);
            
            int pred_c = model.predict_class(user_f);
            double prob_c1 = model.predict_probability_class1(user_f);
            std::string risk_level_string = (pred_c == 1) ? "HIGH" : "LOW";

            std::cout << "\n--- Current Prediction Result ---" << std::endl; 
            std::cout << "Predicted Risk: " << risk_level_string 
                      << (pred_c == 1 ? " (Likely Depressed)" : " (Likely Not Depressed)") << std::endl;
            std::cout << "Probability of High Risk (Depression): " << std::fixed << std::setprecision(3) << prob_c1 << std::endl;

            // Save current prediction to history
            HistoryEntry current_entry;
            current_entry.timestamp = get_current_timestamp();
            current_entry.probability_class1 = prob_c1;
            current_entry.risk_level_str = risk_level_string;
            append_to_risk_history(current_entry);
            std::cout << "Current prediction saved to history." << std::endl;

            display_recommendations(pred_c, user_f, feature_names_list);

        } catch (const std::runtime_error& e) {
             if (std::string(e.what()) == "User quit input.") {
                 std::cout << "Input cancelled by user."<<std::endl; 
                 // Loop will continue to ask "Make a new prediction?"
             } else {
                 std::cerr << "Input error: " << e.what() << std::endl;
             }
        } catch (const std::exception& e) { 
            std::cerr << "An unexpected error occurred: " << e.what() << std::endl;
        }
    }

    std::cout << "\nExiting program." << std::endl;
    return 0;
}