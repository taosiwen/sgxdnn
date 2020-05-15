/*
 * Copyright (C) 2011-2018 Intel Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the
 *     distribution.
 *   * Neither the name of Intel Corporation nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */


#include <stdio.h>
#include <string.h>
#include <assert.h>

# include <unistd.h>
# include <pwd.h>
# define MAX_PATH FILENAME_MAX

#include "sgx_urts.h"
#include "App.h"
#include "Enclave_u.h"

#include <iostream>
#include <Eigen/Eigen>
#include <cstdio>
#include <iomanip>
#include "DNN.cpp"


using namespace std;
using namespace Eigen;

/* Global EID shared by multiple threads */
sgx_enclave_id_t global_eid = 0;



/* Application entry */
int reverse_binary(unsigned char *buffer);
MatrixXf read_image_binary(const char *filename);
MatrixXf read_label_binary(const char *filename);
float ErrorRate(MatrixXf y_out, MatrixXf y_label);
float accucacy(MatrixXf output,MatrixXf label);
MatrixXf convert_eigen(struct MATRIX mat);
struct MATRIX convert_p(MatrixXf mat);
VectorXi Decode(MatrixXf y_out);
/* Application entry */
int SGX_CDECL main(int argc, char *argv[])
{
    //DNN test("SAVE/test.model", "SAVE/test.param");
    

    DNN test;
    Add_SafeLayer(global_eid,784,500);
    test.AddLayer(500,300,LAYER_RELU); 
    test.AddLayer(300,100,LAYER_RELU);
    test.AddLayer(100,10,LAYER_SOFTMAX);


    test.SetLearnRate(0.05);
    test.SetDecayRate(0.002);
    test.SetLossType(LOSS_CROSS_ENTROPY);
    test.SetOptimizeType(OPTIMIZE_MOMENTUM);
   
    test.ShowModel();


    MatrixXf train_image = read_image_binary("/home/sgxsdk/SampleCode/bysj_project/mnist_data/t10k-images.idx3-ubyte");

    MatrixXf train_label = read_label_binary("/home/sgxsdk/SampleCode/bysj_project/mnist_data/train-labels.idx1-ubyte");

    MatrixXf test_image = read_image_binary("/home/sgxsdk/SampleCode/bysj_project/mnist_data/t10k-images.idx3-ubyte");

    MatrixXf test_label = read_label_binary("/home/sgxsdk/SampleCode/bysj_project/mnist_data/t10k-labels.idx1-ubyte");

    
    //for(int i=1; i<=2; i++){ 
    //train
    test.Train_safe(train_image, train_label,200);

    //frontprop
    MatrixXf input = test.GetInputTensor();
    struct MATRIX inp = convert_p(input);
    cout<<inp.row;
    FrontPropSafe(global_eid,inp);
    struct MATRIX safenet;
    get_safe_net(global_eid,&safenet);
    cout<<"========="<<safenet.row;
    test.set_safe_net(convert_eigen(safenet));

    test.FrontPropTotal();


    /*
    //backprop
    test.BackPropTotal();
    MatrixXf weight_api = test.get_first_weight();
    MatrixXf delta_api = test.get_first_delta();
    struct MATRIX *weight = convert_p(weight_api);
    struct MATRIX *delta = convert_p(delta_api);

    BackPropSafe(global_eid,info,*weight,*delta);
    free(weight);
    free(delta);
    //optimize

    test.Optimize(test.GetOptimizeType());
    Opt_safe(global_eid,info,test.GetOptimizeType(),test.GetLearnRate(),test.GetDecayRate());
    float train_loss = test.GetTotalLoss();

    //test

    test.Test_safe(test_image, test_label, 200);
    //frontprop
    MatrixXf input_test = test.GetInputTensor();
    struct MATRIX *inp_test = convert_p(input_test);
    inp_test = &(info->input_tensor);
    FrontPropSafe(global_eid,info,*inp_test);
    struct MATRIX *safenet_test;
    get_safe_net(global_eid,safenet_test,info);
    test.set_safe_net(convert_eigen(safenet_test));
    test.FrontPropTotal();
    float test_loss = test.GetTotalLoss();
    free(inp_test);
    free(safenet_test);

    cout << "[" << i << " epoch]";
    cout << " train : " << setw(8) << train_loss;
    cout << " test : " << setw(8) << test_loss;
    cout << endl;
    

    //total test
    test.Test_safe(test_image, test_label);
    MatrixXf input_all = test.GetInputTensor();
    struct MATRIX *inp_all = convert_p(input_all);
    inp_all = &(info->input_tensor);
    FrontPropSafe(global_eid,info,*inp_all);
    struct MATRIX *safenet_all;
    get_safe_net(global_eid,safenet_all,info);
    test.set_safe_net(convert_eigen(safenet_all));
    test.FrontPropTotal();
    float all_loss = test.GetTotalLoss();
    free(inp_all);
    free(safenet_all);
    
    if(i%100 == 0){
        cout << "Total Test Loss: " << test.GetTotalLoss() << endl;
        cout << "Total Error Rate: " << ErrorRate(test.GetOutputTensor(), test.GetLabelTensor()) << endl;
        cout << "Test Accuracy: "<<accucacy(test.GetOutputTensor(),test.GetLabelTensor()) << endl;
        cout << "Save parameter ..." << endl;
  
    }
    //}
    
    */
    return 0;


}


int reverse_binary(unsigned char *buffer){
    return (int)buffer[3] | (int)buffer[2]<<8 | (int)buffer[1]<<16 | (int)buffer[0]<<24;
}

MatrixXf read_image_binary(const char *filename){
    int mnum;
    int img_total;
    int img_row;
    int img_col;
    
    unsigned char cbuf[4];
    FILE *fp;
    fp = fopen(filename, "rb");
    
    fread(&cbuf, 1, 4, fp);
    mnum = reverse_binary(cbuf);
    fread(&cbuf, 1, 4, fp);
    img_total = reverse_binary(cbuf);
    fread(&cbuf, 1, 4, fp);
    img_row = reverse_binary(cbuf);
    fread(&cbuf, 1, 4, fp);
    img_col = reverse_binary(cbuf);

    cout << mnum << " " << img_total << " " << img_row << " " << img_col << endl;
    //img_total = 200;
    MatrixXf img_buf(img_row*img_col, img_total);
    //cout <<"1"<<endl;
    unsigned char rbuf;
    //cout <<"1"<<endl;
    for(int j=0; j<img_total; ++j){
        //cout <<"1"<<endl;
        for(int i=0; i<img_col*img_row; ++i){
                fread(&rbuf, 1, 1, fp);
                img_buf(i,j) = ((double)rbuf)/255;
        }
    }
    
    return img_buf;
}

MatrixXf read_label_binary(const char *filename){
    int mnum;
    int label_total;
    
    unsigned char cbuf[4];
    FILE *fp;
    fp = fopen(filename, "rb");
    
    fread(&cbuf, 1, 4, fp);
    mnum = reverse_binary(cbuf);
    fread(&cbuf, 1, 4, fp);
    label_total = reverse_binary(cbuf);

    cout << mnum << " " << label_total << endl;
    //label_total = 200;
    MatrixXf label_buf = MatrixXf::Zero(10, label_total);
    unsigned char rbuf;
    for(int i=0; i<label_total; ++i){
        fread(&rbuf, 1, 1, fp);
        label_buf((int)rbuf,i) = 1;
    }
    
    return label_buf;
}

float accucacy(MatrixXf output,MatrixXf label){
    
    return (1-ErrorRate(output,label));
}

float ErrorRate(MatrixXf y_out, MatrixXf y_label){
    int count = 0;
    VectorXi y_decode = Decode(y_out);
    for(int i=0; i<y_label.cols(); ++i)
        if(y_label(y_decode(i), i) != 1)
            ++count;

    return (float)count / y_label.cols(); 
}

VectorXi Decode(MatrixXf y_out){
    VectorXi y_decode(y_out.cols());

    for(int j=0; j<y_out.cols(); ++j){
        float temp = -9999;
        for(int i=0; i<y_out.rows(); ++i){
            if(y_out(i,j) > temp){
                temp = y_out(i,j);
                y_decode(j) = i;
            }
        }
    }

    return y_decode;
}



MatrixXf convert_eigen(struct MATRIX mat){
    MatrixXf temp(mat.row,mat.col);
    for (int i=0;i<mat.row;i++){
        for (int j=0;j<mat.col;j++){
            temp(i,j) = mat.p[i][j];
        }
    }
    return temp;


}

struct MATRIX convert_p(MatrixXf mat)
{
    struct MATRIX matrix;
    matrix.row = mat.rows();
    matrix.col = mat.cols();

    matrix.p = (float**)malloc((matrix.row) * sizeof(float*));
    for (int i = 0; i < matrix.row; i++)
    {
        matrix.p[i] = (float*)malloc((matrix.col) * sizeof(float));
    }
    for (int i=0;i<matrix.row;i++){
        for (int j=0;j<matrix.col;j++){
            matrix.p[i][j] = mat(i,j);
        }
    }
    cout<<matrix.row;
    return matrix;
}