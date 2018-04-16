/*
 * WMY_SPIDNN.h
 *
 *  Created on: 2018年4月12日
 *      Author: wmy
 */

#ifndef SPIDNN_WMY_SPIDNN_H_
#define SPIDNN_WMY_SPIDNN_H_

typedef struct//神经元细胞单体
{
    double INPUT;
    double STATUS;
    double OUTPUT;
}NEURON;

typedef struct
{
    double LEARN_SPEED_W_P_OUT; //学习步长
    double LEARN_SPEED_W_I_OUT; //学习步长
    double LEARN_SPEED_W_D_OUT; //学习步长
    double LEARN_SPEED_W_IN1_P; //学习步长
    double LEARN_SPEED_W_IN1_I; //学习步长
    double LEARN_SPEED_W_IN1_D; //学习步长
    double LEARN_SPEED_W_IN2_P; //学习步长
    double LEARN_SPEED_W_IN2_I; //学习步长
    double LEARN_SPEED_W_IN2_D; //学习步长
    int LEARN_COUNT;//学习次数 L
    double W_IN1_P;    // Init -1 NOW 权重值
    double W_IN1_I;    // Init -1 NOW 权重值
    double W_IN1_D;    // Init -1 NOW 权重值
    double W_IN2_P;    // Init 1 SET 权重值
    double W_IN2_I;    // Init 1 SET 权重值
    double W_IN2_D;    // Init 1 SET 权重值
    double W_P_OUT;    //权重值
    double W_I_OUT;    //权重值
    double W_D_OUT;    //权重值
    NEURON INPUT_1_NOW_POINT;//输入层
    NEURON INPUT_2_SET_POINT;//输入层
    NEURON P_neuron;//隐含层
    NEURON I_neuron;//隐含层
    NEURON D_neuron;//隐含层
    NEURON OUTPUT;//输出层
    double D_neuron_LAST_INPUT;//微分神经元上一次输入值
    double Sigma_Error_Square;
    double Sigma_W_P_OUT;//-L*(E对W_P_OUT求偏导数）
    double Sigma_W_I_OUT;
    double Sigma_W_D_OUT;
    double Sigma_W_IN1_P;
    double Sigma_W_IN1_I;
    double Sigma_W_IN1_D;
    double Sigma_W_IN2_P;
    double Sigma_W_IN2_I;
    double Sigma_W_IN2_D;
    double LAST_P_INPUT;
    double LAST_I_INPUT;
    double LAST_D_INPUT;
    double LAST_P_STATUS;
    double LAST_I_STATUS;
    double LAST_D_STATUS;
    double LAST_SPIDNN_FORWARD_RETURN;
    double NOW_SPIDNN_FORWARD_RETURN;
    double Learn_Speed_W_P_OUT_Coefficient;
    double Learn_Speed_W_I_OUT_Coefficient;
    double Learn_Speed_W_D_OUT_Coefficient;
    double Learn_Speed_W_IN1_P_Coefficient;
    double Learn_Speed_W_IN1_I_Coefficient;
    double Learn_Speed_W_IN1_D_Coefficient;
    double Learn_Speed_W_IN2_P_Coefficient;
    double Learn_Speed_W_IN2_I_Coefficient;
    double Learn_Speed_W_IN2_D_Coefficient;
}SPIDNN;

extern double NEURON_OUTPUT_FUNCTION(double x);
extern int Sign(double x);
extern double SPIDNN_FORWARD_CONTROL(SPIDNN * spidnn, double setpoint, double nowpoint);//神经元网络PID前向控制算法
extern void SPIDNN_BACK_CONTROL(SPIDNN * spidnn, double setpoint, double nowpoint);//神经元网络PID反向传播算法

#endif /* SPIDNN_WMY_SPIDNN_H_ */
