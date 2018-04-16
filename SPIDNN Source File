/*
 * WMY_SPIDNN.c
 *
 *  Created on: 2018年4月10日
 *      Author: wmy
 */

#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include "inc/hw_memmap.h"
#include "driverlib/fpu.h"
#include "inc/hw_types.h"
#include "driverlib/sysctl.h"
#include "driverlib/gpio.h"

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

double NEURON_OUTPUT_FUNCTION(double x)//函数形式不唯一
{
    return x;
}

int Sign(double x)//符号函数
{
    if(x>0)
    {
        return 1;
    }
    else if(x<0)
    {
        return -1;
    }
    return 0;
}

double SPIDNN_FORWARD_CONTROL(SPIDNN * spidnn, double setpoint, double nowpoint)//神经元网络PID前向控制算法
{
    double SPIDNN_FORWARD_CONTROL_FINAL_OUTPUT;
    spidnn->Sigma_Error_Square = spidnn->Sigma_Error_Square+pow((setpoint-nowpoint),2);
    spidnn->LAST_SPIDNN_FORWARD_RETURN = spidnn->NOW_SPIDNN_FORWARD_RETURN;
    spidnn->LAST_P_INPUT = spidnn->P_neuron.INPUT;
    spidnn->LAST_I_INPUT = spidnn->I_neuron.INPUT;
    spidnn->LAST_D_INPUT = spidnn->D_neuron.INPUT;
    spidnn->LAST_P_STATUS = spidnn->P_neuron.STATUS;
    spidnn->LAST_I_STATUS = spidnn->I_neuron.STATUS;
    spidnn->LAST_D_STATUS = spidnn->D_neuron.STATUS;
    spidnn->INPUT_1_NOW_POINT.INPUT = nowpoint;
    spidnn->INPUT_2_SET_POINT.INPUT = setpoint;
    spidnn->INPUT_1_NOW_POINT.STATUS=spidnn->INPUT_1_NOW_POINT.INPUT;
    spidnn->INPUT_2_SET_POINT.STATUS = spidnn->INPUT_2_SET_POINT.INPUT;
    spidnn->INPUT_1_NOW_POINT.OUTPUT = NEURON_OUTPUT_FUNCTION(spidnn->INPUT_1_NOW_POINT.STATUS);
    spidnn->INPUT_2_SET_POINT.OUTPUT = NEURON_OUTPUT_FUNCTION(spidnn->INPUT_2_SET_POINT.STATUS);
    spidnn->P_neuron.INPUT = spidnn->W_IN1_P* spidnn->INPUT_1_NOW_POINT.OUTPUT + spidnn->W_IN2_P *spidnn->INPUT_2_SET_POINT.OUTPUT;
    spidnn->P_neuron.STATUS = spidnn->P_neuron.INPUT;
    spidnn->P_neuron.OUTPUT= NEURON_OUTPUT_FUNCTION(spidnn->P_neuron.STATUS);
    spidnn->I_neuron.INPUT = spidnn->W_IN1_I * spidnn->INPUT_1_NOW_POINT.OUTPUT + spidnn->W_IN2_I * spidnn->INPUT_2_SET_POINT.OUTPUT;
    spidnn->I_neuron.STATUS = spidnn->I_neuron.STATUS + spidnn->I_neuron.INPUT;
    spidnn->I_neuron.OUTPUT= NEURON_OUTPUT_FUNCTION( spidnn->I_neuron.STATUS );
    spidnn->D_neuron.INPUT= spidnn->W_IN1_D * spidnn->INPUT_1_NOW_POINT.OUTPUT+spidnn->W_IN2_D * spidnn->INPUT_2_SET_POINT.OUTPUT;
    spidnn->D_neuron.STATUS = spidnn->D_neuron.INPUT - spidnn->D_neuron_LAST_INPUT;
    spidnn->D_neuron.OUTPUT= NEURON_OUTPUT_FUNCTION(spidnn->D_neuron.STATUS);
    spidnn->D_neuron_LAST_INPUT = spidnn->D_neuron.INPUT ;
    spidnn->OUTPUT.INPUT = spidnn->W_P_OUT * spidnn->P_neuron.OUTPUT + spidnn->W_I_OUT *  spidnn->I_neuron.OUTPUT + spidnn->W_D_OUT * spidnn->D_neuron.OUTPUT;
    spidnn->OUTPUT.STATUS = spidnn->OUTPUT.INPUT;
    spidnn->OUTPUT.OUTPUT = NEURON_OUTPUT_FUNCTION( spidnn->OUTPUT.STATUS);
    SPIDNN_FORWARD_CONTROL_FINAL_OUTPUT = spidnn->OUTPUT.OUTPUT;
    spidnn->NOW_SPIDNN_FORWARD_RETURN=SPIDNN_FORWARD_CONTROL_FINAL_OUTPUT;
    return SPIDNN_FORWARD_CONTROL_FINAL_OUTPUT;
}

void SPIDNN_BACK_CONTROL(SPIDNN * spidnn, double setpoint, double nowpoint)//神经元网络PID反向传播算法
{
    spidnn->LEARN_COUNT ++;//学习次数+1
    //**********P_OUT**********
    spidnn->Sigma_W_P_OUT = spidnn->Sigma_W_P_OUT + (2*(spidnn->INPUT_2_SET_POINT.INPUT-spidnn->INPUT_1_NOW_POINT.INPUT)*
    Sign((nowpoint-spidnn->INPUT_1_NOW_POINT.INPUT)/(spidnn->NOW_SPIDNN_FORWARD_RETURN-spidnn->LAST_SPIDNN_FORWARD_RETURN)))*spidnn->P_neuron.OUTPUT;
    spidnn->LEARN_SPEED_W_P_OUT=spidnn->Learn_Speed_W_P_OUT_Coefficient/ pow((-(spidnn->Sigma_W_P_OUT/spidnn->LEARN_COUNT)/sqrt((spidnn->Sigma_Error_Square/spidnn->LEARN_COUNT))),2);
    //**********I_OUT**********
    spidnn->Sigma_W_I_OUT = spidnn->Sigma_W_I_OUT + (2*(spidnn->INPUT_2_SET_POINT.INPUT-spidnn->INPUT_1_NOW_POINT.INPUT)*
    Sign((nowpoint-spidnn->INPUT_1_NOW_POINT.INPUT)/(spidnn->NOW_SPIDNN_FORWARD_RETURN-spidnn->LAST_SPIDNN_FORWARD_RETURN)))*spidnn->I_neuron.OUTPUT;
    spidnn->LEARN_SPEED_W_I_OUT=spidnn->Learn_Speed_W_I_OUT_Coefficient/ pow((-(spidnn->Sigma_W_I_OUT/spidnn->LEARN_COUNT)/sqrt((spidnn->Sigma_Error_Square/spidnn->LEARN_COUNT))),2);
    //**********D_OUT**********
    spidnn->Sigma_W_D_OUT = spidnn->Sigma_W_D_OUT + (2*(spidnn->INPUT_2_SET_POINT.INPUT-spidnn->INPUT_1_NOW_POINT.INPUT)*
    Sign((nowpoint-spidnn->INPUT_1_NOW_POINT.INPUT)/(spidnn->NOW_SPIDNN_FORWARD_RETURN-spidnn->LAST_SPIDNN_FORWARD_RETURN)))*spidnn->D_neuron.OUTPUT;
    spidnn->LEARN_SPEED_W_D_OUT=spidnn->Learn_Speed_W_D_OUT_Coefficient/ pow((-(spidnn->Sigma_W_D_OUT/spidnn->LEARN_COUNT)/sqrt((spidnn->Sigma_Error_Square/spidnn->LEARN_COUNT))),2);
    //**********IN1_P**********
    spidnn->Sigma_W_IN1_P =  spidnn->Sigma_W_IN1_P +  ((2*(spidnn->INPUT_2_SET_POINT.INPUT-spidnn->INPUT_1_NOW_POINT.INPUT)*
    Sign((nowpoint-spidnn->INPUT_1_NOW_POINT.INPUT)/(spidnn->NOW_SPIDNN_FORWARD_RETURN-spidnn->LAST_SPIDNN_FORWARD_RETURN)))*spidnn->W_P_OUT*
    Sign((spidnn->P_neuron.STATUS-spidnn->LAST_P_STATUS)/(spidnn->P_neuron.INPUT-spidnn->LAST_P_INPUT)))*spidnn->INPUT_1_NOW_POINT.OUTPUT;
    spidnn->LEARN_SPEED_W_IN1_P = spidnn->Learn_Speed_W_IN1_P_Coefficient/ pow((-(spidnn->Sigma_W_IN1_P/spidnn->LEARN_COUNT)/sqrt((spidnn->Sigma_Error_Square/spidnn->LEARN_COUNT))),2);
    //**********IN1_I**********
    spidnn->Sigma_W_IN1_I =  spidnn->Sigma_W_IN1_I +  ((2*(spidnn->INPUT_2_SET_POINT.INPUT-spidnn->INPUT_1_NOW_POINT.INPUT)*
    Sign((nowpoint-spidnn->INPUT_1_NOW_POINT.INPUT)/(spidnn->NOW_SPIDNN_FORWARD_RETURN-spidnn->LAST_SPIDNN_FORWARD_RETURN)))*spidnn->W_I_OUT*
    Sign((spidnn->I_neuron.STATUS-spidnn->LAST_I_STATUS)/(spidnn->I_neuron.INPUT-spidnn->LAST_I_INPUT)))*spidnn->INPUT_1_NOW_POINT.OUTPUT;
    spidnn->LEARN_SPEED_W_IN1_I = spidnn->Learn_Speed_W_IN1_I_Coefficient/ pow((-(spidnn->Sigma_W_IN1_I/spidnn->LEARN_COUNT)/sqrt((spidnn->Sigma_Error_Square/spidnn->LEARN_COUNT))),2);
    //**********IN1_D**********
    spidnn->Sigma_W_IN1_D =  spidnn->Sigma_W_IN1_D +  ((2*(spidnn->INPUT_2_SET_POINT.INPUT-spidnn->INPUT_1_NOW_POINT.INPUT)*
    Sign((nowpoint-spidnn->INPUT_1_NOW_POINT.INPUT)/(spidnn->NOW_SPIDNN_FORWARD_RETURN-spidnn->LAST_SPIDNN_FORWARD_RETURN)))*spidnn->W_D_OUT*
    Sign((spidnn->D_neuron.STATUS-spidnn->LAST_D_STATUS)/(spidnn->D_neuron.INPUT-spidnn->LAST_D_INPUT)))*spidnn->INPUT_1_NOW_POINT.OUTPUT;
    spidnn->LEARN_SPEED_W_IN1_D = spidnn->Learn_Speed_W_IN1_D_Coefficient/ pow((-(spidnn->Sigma_W_IN1_D/spidnn->LEARN_COUNT)/sqrt((spidnn->Sigma_Error_Square/spidnn->LEARN_COUNT))),2);
    //**********IN2_P**********
    spidnn->Sigma_W_IN2_P =  spidnn->Sigma_W_IN2_P +  ((2*(spidnn->INPUT_2_SET_POINT.INPUT-spidnn->INPUT_1_NOW_POINT.INPUT)*
    Sign((nowpoint-spidnn->INPUT_1_NOW_POINT.INPUT)/(spidnn->NOW_SPIDNN_FORWARD_RETURN-spidnn->LAST_SPIDNN_FORWARD_RETURN)))*spidnn->W_P_OUT*
    Sign((spidnn->P_neuron.STATUS-spidnn->LAST_P_STATUS)/(spidnn->P_neuron.INPUT-spidnn->LAST_P_INPUT)))*spidnn->INPUT_2_SET_POINT.OUTPUT;
    spidnn->LEARN_SPEED_W_IN2_P = spidnn->Learn_Speed_W_IN2_P_Coefficient/ pow((-(spidnn->Sigma_W_IN2_P/spidnn->LEARN_COUNT)/sqrt((spidnn->Sigma_Error_Square/spidnn->LEARN_COUNT))),2);
    //**********IN2_I**********
    spidnn->Sigma_W_IN2_I =  spidnn->Sigma_W_IN2_I +  ((2*(spidnn->INPUT_2_SET_POINT.INPUT-spidnn->INPUT_1_NOW_POINT.INPUT)*
    Sign((nowpoint-spidnn->INPUT_1_NOW_POINT.INPUT)/(spidnn->NOW_SPIDNN_FORWARD_RETURN-spidnn->LAST_SPIDNN_FORWARD_RETURN)))*spidnn->W_I_OUT*
    Sign((spidnn->I_neuron.STATUS-spidnn->LAST_I_STATUS)/(spidnn->I_neuron.INPUT-spidnn->LAST_I_INPUT)))*spidnn->INPUT_2_SET_POINT.OUTPUT;
    spidnn->LEARN_SPEED_W_IN2_I = spidnn->Learn_Speed_W_IN2_I_Coefficient/ pow((-(spidnn->Sigma_W_IN2_I/spidnn->LEARN_COUNT)/sqrt((spidnn->Sigma_Error_Square/spidnn->LEARN_COUNT))),2);
    //**********IN2_D**********
    spidnn->Sigma_W_IN2_D =  spidnn->Sigma_W_IN2_D +  ((2*(spidnn->INPUT_2_SET_POINT.INPUT-spidnn->INPUT_1_NOW_POINT.INPUT)*
    Sign((nowpoint-spidnn->INPUT_1_NOW_POINT.INPUT)/(spidnn->NOW_SPIDNN_FORWARD_RETURN-spidnn->LAST_SPIDNN_FORWARD_RETURN)))*spidnn->W_D_OUT*
    Sign((spidnn->D_neuron.STATUS-spidnn->LAST_D_STATUS)/(spidnn->D_neuron.INPUT-spidnn->LAST_D_INPUT)))*spidnn->INPUT_2_SET_POINT.OUTPUT;
    spidnn->LEARN_SPEED_W_IN2_D = spidnn->Learn_Speed_W_IN2_D_Coefficient/ pow((-(spidnn->Sigma_W_IN2_D/spidnn->LEARN_COUNT)/sqrt((spidnn->Sigma_Error_Square/spidnn->LEARN_COUNT))),2);
    //修改比例神经元至输出神经元的权重值
    spidnn->W_P_OUT = spidnn->W_P_OUT + (spidnn->LEARN_SPEED_W_P_OUT/spidnn->LEARN_COUNT)*spidnn->Sigma_W_P_OUT;
    //修改积分神经元至输出神经元的权重值
    spidnn->W_I_OUT = spidnn->W_I_OUT + (spidnn->LEARN_SPEED_W_I_OUT/spidnn->LEARN_COUNT)*spidnn->Sigma_W_I_OUT;
    //修改微分神经元至输出神经元的权重值
    spidnn->W_D_OUT = spidnn->W_D_OUT + (spidnn->LEARN_SPEED_W_D_OUT/spidnn->LEARN_COUNT)*spidnn->Sigma_W_D_OUT;
    //修改输入1神经元至比例神经元的权重值
    spidnn->W_IN1_P = spidnn->W_IN1_P + (spidnn->LEARN_SPEED_W_IN1_P/spidnn->LEARN_COUNT)*spidnn->Sigma_W_IN1_P;
    //修改输入1神经元至积分神经元的权重值
    spidnn->W_IN1_I = spidnn->W_IN1_I + (spidnn->LEARN_SPEED_W_IN1_I/spidnn->LEARN_COUNT)*spidnn->Sigma_W_IN1_I;
    //修改输入1神经元至微分神经元的权重值
    spidnn->W_IN1_D = spidnn->W_IN1_D + (spidnn->LEARN_SPEED_W_IN1_D/spidnn->LEARN_COUNT)*spidnn->Sigma_W_IN1_D;
    //修改输入2神经元至比例神经元的权重值
    spidnn->W_IN2_P = spidnn->W_IN2_P + (spidnn->LEARN_SPEED_W_IN2_P/spidnn->LEARN_COUNT)*spidnn->Sigma_W_IN2_P;
    //修改输入2神经元至积分神经元的权重值
    spidnn->W_IN2_I = spidnn->W_IN2_I + (spidnn->LEARN_SPEED_W_IN2_I/spidnn->LEARN_COUNT)*spidnn->Sigma_W_IN2_I;
    //修改输入2神经元至微分神经元的权重值
    spidnn->W_IN2_D = spidnn->W_IN2_D + (spidnn->LEARN_SPEED_W_IN2_D/spidnn->LEARN_COUNT)*spidnn->Sigma_W_IN2_D;
}

