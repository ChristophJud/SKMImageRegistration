/*
 * Copyright 2016 University of Basel, Medical Image Analysis Center
 *
 * Author: Christoph Jud     (christoph.jud@unibas.ch)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 
#pragma once

#include <fstream>
#include <algorithm>

#include "json.hpp"
using json = nlohmann::json;

#include "table_printer.h"

json load_json_config(){
    json config = {
        {"reference_filename",      "/tmp/ref.vtk"},
        {"target_filename",         "/tmp/tar.vtk"},
        {"reference_landmarks",     "none"},
        {"target_landmarks",        "none"},
        {"weight_filename",         "none"},
        {"weight_tensor_filename",  "none"},
        {"temp_directory",          "/tmp/skm/"},
        {"sigma_C4",                {64, 32, 16}},
        {"sigma_C0",                {128, 64, 32}},
        {"cp_spacing_scaler",       4.0},
        {"tensor_weights",          {400, 200, 100}},
        {"reg_l1",                  {0, 0, 0}},
        {"reg_l21",                 {0, 0, 0}},
        {"reg_l2",                  {0, 0, 0}},
        {"reg_rkhs",                {0, 0, 0}},
        {"reg_rd",                  {0, 0, 0}},
        {"reg_rd_scaling",          {1, 1, 1}},
        {"reg_pg",                  {0, 0, 0}},
        {"reg_pg_scaling",          {1, 1, 1}},
        {"geometric_decay",         1},
        {"num_scales",              3},
        {"restart_at",              0},
        {"num_function_evaluations",{500, 500, 500}},
        {"initial_step_size",       {0.1, 0.1, 0.1}},
        {"metric",                  "mse"},
        {"sampling_rate",           {5, 4, 3}},
        {"neighborhood_sampling",   {15, 12, 9}},
        {"intensity_thresholds",    {-1500, 3500}},
        {"smoothing_variance",      {2, 1, 0}},
        {"verbosity",               1}                
    };
    return config;
}

json load_json_config(std::string config_filename){
    // remove quotation marks
    char chars[] = "\"";
    for(auto c:chars) config_filename.erase( std::remove(config_filename.begin(), config_filename.end(), c), config_filename.end() );

    std::ifstream ifs(config_filename, std::ifstream::in);
    json config;
    if(ifs.is_open()){
        ifs >> config;
    }
    return config;
}

class InputParser{
    public:
        InputParser (int &argc, char **argv){
            for (int i=1; i < argc; ++i){
                std::string argument = argv[i];

                // if argument or value contains slashes add ""
                std::stringstream ss;
                auto found = argument.find('/');
                if(found!=std::string::npos && argument[0] != '"')
                    ss << '"' << argument << '"';
                else
                    ss << argument;

                this->tokens.push_back(ss.str());
            }
        }
        /// @author iain
        const std::string& getCmdOption(const std::string &option) const{
            std::vector<std::string>::const_iterator itr;
            itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
            if (itr != this->tokens.end() && ++itr != this->tokens.end()){
                return *itr;
            }
            static const std::string empty_string("");
            return empty_string;
        }
        /// @author iain
        bool cmdOptionExists(const std::string &option) const{
            return std::find(this->tokens.begin(), this->tokens.end(), option)
                   != this->tokens.end();
        }
    private:
        std::vector <std::string> tokens;
};

void print_json_config(const json& config){

    unsigned max_str_length = 0;
    unsigned max_val_length = 0;
    for(json::const_iterator item = config.begin(); item!=config.end(); ++item){
        if(item.key().size()>max_str_length){
            max_str_length = item.key().size();
        }

        std::stringstream ss; ss << item.value();
        if(ss.str().size()>max_val_length){
            max_val_length = ss.str().size();
        }
    }

    const std::string par_string("Parameter");
    const std::string val_string("Value");

    bprinter::TablePrinter tp(&std::cout);
    tp.set_flush_left();
    tp.AddColumn(par_string, std::max(static_cast<unsigned>(par_string.size()),max_str_length));
    tp.AddColumn(val_string,    std::max(static_cast<unsigned>(val_string.size()),max_val_length));
    tp.PrintHeader();

    for(json::const_iterator item = config.begin(); item!=config.end(); ++item){
        std::stringstream ss; ss << item.value();
        tp << item.key() <<  ss.str();
    }
    tp.PrintFooter();

    return;
}