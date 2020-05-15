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

#include "Enclave.h"
#include "Enclave_t.h" /* print_string */
#include <cstdio> /* vsnprintf */
#include <string.h>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>
#include <iterator>
#include <stdarg.h>
using namespace std;
#define MOMENTUM_ALPHA 0.2
/* 
 * printf: 
 *   Invokes OCALL to display the enclave buffer to the terminal.
 */

/*
typedef struct safeLayerInfo{
    int input;
    int output;

    struct MATRIX weight;
    struct MATRIX safe_weight_rec;
    struct MATRIX safe_net;
    struct MATRIX safe_delta;
    struct MATRIX safe_weight_delta;
    struct MATRIX input_tensor;

}safeLayerInfo;

typedef struct MATRIX{
	float **p;
	int row;
	int col;
}MATRIX;

*/

//float Gaussrand(float exp, float var);

static struct safeLayerInfo safe_info;
float **zero_matrix(int row,int col); 
struct MATRIX multi(struct MATRIX a,struct MATRIX b);
struct MATRIX multi_by_value(struct MATRIX a,float value);
struct MATRIX add(struct MATRIX a,struct MATRIX b);
struct MATRIX transpose(struct MATRIX a);
struct MATRIX initialize(int row,int col);
//void printf(const char *fmt,...);

void Add_SafeLayer(unsigned int input,unsigned int output){
	safe_info.input = input;
	safe_info.output = output;

	//initialize weight
	//srand((unsigned int)time(0));
	int rows = safe_info.output;
	int cols = safe_info.input;
	int num= 6;

	safe_info.weight = initialize(rows,cols);

	const int non_zero = rows * num;

	int row[non_zero];
	for(int i=0;i<non_zero;i++){
		row[i] = i%rows;
	}
	int col[non_zero];
	for(int i=0;i<non_zero;i++){
		//col[i] = rand() % cols;
		col[i] = i % cols;
	}
	float val[non_zero];
	for(int i=0;i<non_zero;i++){
		//val[i] = Gaussrand(0, 1/sqrt(rows));
		val[i] = 0;
	}
	
	for (int i=0;i<non_zero;i++){
		(safe_info.weight).p[row[i]][col[i]] = val[i];
	} //sparse matrix

	//initialize weight_rec
	safe_info.safe_weight_rec = initialize(rows,cols);
	//printf(safe_info.input);
}

float **zero_matrix(int row,int col)
{
	float **temp = (float**)malloc(row * sizeof(float*));

 	for (int i = 0; i < row; i++)
	{
		temp[i] = (float*)malloc(col * sizeof(float));
	}
	for (int i=0;i<row;i++){
		for (int j=0;j<col;j++){
			temp[i][j] = 0;
		}
	}
	return temp;
}

struct MATRIX initialize(int row,int col){
	struct MATRIX temp;
	temp.row = row;
	temp.col = col;
	temp.p = zero_matrix(row,col);
	return temp;
		
}

void FrontPropSafe(struct MATRIX input)
{	
	//return  safenet
	safe_info.input_tensor = input;
	safe_info.safe_net = multi(safe_info.weight,safe_info.input_tensor);

}

void BackPropSafe(struct MATRIX weight,struct MATRIX delta){

	struct MATRIX safe_delta = multi(transpose(weight),delta);
	struct MATRIX safe_weight_delta = initialize((safe_info.weight).row,(safe_info.weight).col);

	for (int i=0;i<(safe_info.input_tensor).col;i++){
		struct MATRIX temp = initialize((safe_info.weight).row,(safe_info.weight).col);
		for (int q=0;q<temp.row;q++){
			for(int j=0;j<temp.col;j++){
				temp.p[q][j] = safe_delta.p[q][i] * (transpose(safe_info.input_tensor)).p[i][j];
				//safe_weight_delta += safe_delta.col(i) * inp.col(i).transpose();
			}
		}
		safe_weight_delta = add(safe_weight_delta,temp);	
	}
	for (int i=0;i<(safe_info.weight).row;i++){
		for (int j=0;j<(safe_info.weight).col;j++){
			safe_weight_delta.p[i][j] /= i<(safe_info.input_tensor).col;
		}
	}
	//safe_weight_delta /=inp.cols();

	safe_info.safe_delta = safe_delta;
	safe_info.safe_weight_delta = safe_weight_delta;

}

void Opt_safe(int type,float learn_rate,float decay_rate){

	switch(type){
		case 0:
			for(int i=0;i<(safe_info.weight).row;i++){
				for(int j=0;j<(safe_info.weight).col;j++){
					if ((safe_info.weight).p[i][j]!=0)
						(safe_info.weight).p[i][j] -= learn_rate*((safe_info.safe_weight_delta).p[i][j]+decay_rate*((safe_info.weight).p[i][j]));
				}
			}
			break;

		case 1:
			float alpha = MOMENTUM_ALPHA;
			struct MATRIX weight_learn = add(safe_info.safe_weight_delta,multi_by_value(safe_info.weight,decay_rate));

			safe_info.safe_weight_rec = add(weight_learn,multi_by_value(safe_info.safe_weight_rec,alpha));
			
			for(int i=0;i<(safe_info.weight).row;i++){
				for(int j=0;j<(safe_info.weight).col;j++){
					if ((safe_info.weight).p[i][j]!=0)
						(safe_info.weight).p[i][j] -= learn_rate*(safe_info.safe_weight_rec).p[i][j];	
				}
			}
	
			break;
	}

}

struct MATRIX get_safe_net(void){
	return(safe_info.safe_net);
}

/*
float Gaussrand(float exp, float var){
	static float V1, V2, S;
	static int phase = 0;
	float X;

	if(phase == 0){
		do {
			float U1 = (float)rand() / RAND_MAX;
			float U2 = (float)rand() / RAND_MAX;
 
			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
		} while(S >= 1 || S == 0);
		
		X = V1 * sqrt(-2 * log(S) / S);
	}
	else
		X = V2 * sqrt(-2 * log(S) / S);
	
	phase = 1 - phase;
	X = X * var + exp;
	return X;
}
*/

struct MATRIX multi(struct MATRIX a,struct MATRIX b){
	//if (a.col != b.row)
	//	exit(0);
	struct MATRIX temp;
	temp.row = a.row;
	temp.col = b.col;
	temp.p = (float**)malloc(temp.row * sizeof(float*));

 	for (int i = 0; i < temp.row; i++)
	{
		temp.p[i] = (float*)malloc(temp.col * sizeof(float));
	}
	for (int i=0;i<temp.row;i++){
		for (int j=0;j<temp.col;j++){
			float sum =0;
			for (int q = 0;q<a.col;q++){
				sum+=a.p[i][q]*b.p[q][j];
			}
			temp.p[i][j] = sum;
		}
	}
	return temp;

}

struct MATRIX add(struct MATRIX a,struct MATRIX b){
	//if ((a.col != b.col)||(a.row !=b.row))
	//	exit(0);
	struct MATRIX temp;
	temp.row = a.row;
	temp.col = a.col;
	temp.p = (float**)malloc(temp.row * sizeof(float*));
	for (int i = 0; i < temp.row; i++)
	{
		temp.p[i] = (float*)malloc(temp.col * sizeof(float));
	}
	for (int i=0;i<temp.row;i++){
		for (int j=0;j<temp.col;j++){
			temp.p[i][j] = a.p[i][j]+b.p[i][j];
		}
	}
	return temp;

}

struct MATRIX transpose(struct MATRIX a){
	struct MATRIX temp;
	temp.row = a.col;
	temp.col = a.row;
	temp.p = (float**)malloc(temp.row * sizeof(float*));
	for (int i = 0; i < temp.row; i++)
	{
		temp.p[i] = (float*)malloc(temp.col * sizeof(float));
	}
	for (int i=0;i<temp.row;i++){
		for (int j=0;j<temp.col;j++){
			temp.p[i][j] = a.p[j][i];
		}
	}
	return temp;
}

struct MATRIX multi_by_value(struct MATRIX a,float value){
	struct MATRIX temp = a;
	for (int i=0;i<a.row;i++){
		for (int j=0;j<a.col;j++){
			a.p[i][j] *= value;
		}
	}
	return temp;
}

/*
void printf(const char *fmt,...){
	char buf[BUFSIZ] = {'\0'};
	va_list ap;
	va_start(ap,fmt);
	vsnprintf(buf,BUFSIZ,fmt,ap);
	va_end(ap);
	ocall_print_string(buf);
}
*/