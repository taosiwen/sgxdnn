typedef struct MATRIX{
    float **p;
    int row;
    int col;
}MATRIX;

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