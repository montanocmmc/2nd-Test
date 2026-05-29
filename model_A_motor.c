#include <stdint.h>
#include <stdbool.h>
#include "inc/hw_ints.h"
#include "inc/hw_memmap.h"
#include "inc/hw_types.h"
#include "driverlib/debug.h"
#include "driverlib/fpu.h"
#include "driverlib/gpio.h"
#include "driverlib/interrupt.h"
#include "driverlib/pin_map.h"
#include "driverlib/sysctl.h"
#include "driverlib/timer.h"
#include "driverlib/uart.h"
#include "utils/uartstdio.h"
#include "utils/uartstdio.c"
#include "driverlib/adc.h"
#include "driverlib/pwm.h"
#include "driverlib/rom.h"
#include "driverlib/rom_map.h"


#ifdef DEBUG
void
__error__(char *pcFilename, uint32_t ui32Line)
{
}
#endif

void setup_gpio();
void setup_timer();
void setup_uart();
void setup_pwm();

void motor_start(uint8_t duty);
void motor_stop();

void timer0A_handler(void);

uint32_t g_ui32SysClock;

char comando = 'X';
bool estado_led = false;
char data[100];

int main(void)
{
    g_ui32SysClock = MAP_SysCtlClockFreqSet((SYSCTL_XTAL_25MHZ |
                                             SYSCTL_OSC_MAIN |
                                             SYSCTL_USE_PLL |
                                             SYSCTL_CFG_VCO_240), 120000000);

    setup_gpio();

    setup_timer();

    setup_uart();

    setup_pwm();

    while(1)
    {
        UARTgets(data, 20);

        char nivel = data[0];  
        UARTprintf("Nivel recibido:%c\n", nivel);

        if(nivel == 'C') 
        {
            comando = 'C';
            TimerLoadSet(TIMER0_BASE, TIMER_A, g_ui32SysClock); 
        }
        else if(nivel == 'S') 
        {
            comando = 'S';
            TimerLoadSet(TIMER0_BASE, TIMER_A, g_ui32SysClock);
        }
        else if(nivel == 'M')  
        {
            comando = 'X'; 
            TimerLoadSet(TIMER0_BASE, TIMER_A, g_ui32SysClock*3);
            motor_start(50);  
        }
    }
}

void setup_gpio() {

    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOB);
    while(!SysCtlPeripheralReady(SYSCTL_PERIPH_GPIOB))
    {
    }

    GPIOPinTypeGPIOOutput(GPIO_PORTB_BASE, 0x30);

    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOE);
    while(!SysCtlPeripheralReady(SYSCTL_PERIPH_GPIOE))
    {
    }

    GPIOPinTypeGPIOOutput(GPIO_PORTE_BASE, 0x03);
}

void setup_timer() {

    SysCtlPeripheralEnable(SYSCTL_PERIPH_TIMER0);
    while(!SysCtlPeripheralReady(SYSCTL_PERIPH_TIMER0))
    {
    }

    IntMasterEnable();

    TimerConfigure(TIMER0_BASE, TIMER_CFG_PERIODIC);
    TimerLoadSet(TIMER0_BASE, TIMER_A, g_ui32SysClock*3);

    IntEnable(INT_TIMER0A);
    TimerIntEnable(TIMER0_BASE, TIMER_TIMA_TIMEOUT);
    TimerEnable(TIMER0_BASE, TIMER_A);
}

void timer0A_handler(void)
{
    TimerIntClear(TIMER0_BASE, TIMER_TIMA_TIMEOUT);

    estado_led = !estado_led; 

    if(comando == 'C') 
    {
        GPIOPinWrite(GPIO_PORTB_BASE, 0x20, 0x00);

        if(estado_led == true) {
            GPIOPinWrite(GPIO_PORTB_BASE, 0x10, 0x10); 
        } else {
            GPIOPinWrite(GPIO_PORTB_BASE, 0x10, 0x00);         
        } 
    }
    else if(comando == 'S') 
    {
        GPIOPinWrite(GPIO_PORTB_BASE, 0x10, 0x00);

        if(estado_led == true) {
            GPIOPinWrite(GPIO_PORTB_BASE, 0x20, 0x20); 
        } else {
            GPIOPinWrite(GPIO_PORTB_BASE, 0x20, 0x00);          
        }
    }
    else 
    {
        if(estado_led == true) {
            GPIOPinWrite(GPIO_PORTB_BASE, 0x30, 0x30); 
        } else {
            GPIOPinWrite(GPIO_PORTB_BASE, 0x30, 0x00);                    
        }
    }
}

void setup_uart() {

    SysCtlPeripheralEnable(SYSCTL_PERIPH_UART0);
    while(!SysCtlPeripheralReady(SYSCTL_PERIPH_UART0)) {}

    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOA);
    while(!SysCtlPeripheralReady(SYSCTL_PERIPH_GPIOA)) {}
    GPIOPinConfigure(GPIO_PA0_U0RX);
    GPIOPinConfigure(GPIO_PA1_U0TX);
    GPIOPinTypeUART(GPIO_PORTA_BASE, 0X03);

    UARTStdioConfig(0,9600,120000000);

}

void setup_pwm() {

    SysCtlPeripheralEnable(SYSCTL_PERIPH_PWM0);
    while(!SysCtlPeripheralReady(SYSCTL_PERIPH_PWM0)) {}
    
    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOK);
    while(!SysCtlPeripheralReady(SYSCTL_PERIPH_GPIOK)) {}
    
    GPIOPinConfigure(GPIO_PK5_M0PWM7);
    GPIOPinTypePWM(GPIO_PORTK_BASE, GPIO_PIN_5);
    
    PWMGenConfigure(PWM0_BASE, PWM_GEN_3, PWM_GEN_MODE_DOWN);
    
    PWMGenPeriodSet(PWM0_BASE, PWM_GEN_3, 4800);
    
    PWMPulseWidthSet(PWM0_BASE, PWM_OUT_7, 1);
    
    PWMGenEnable(PWM0_BASE, PWM_GEN_3);
    
    PWMOutputState(PWM0_BASE, PWM_OUT_7_BIT, false);

}

void motor_start(uint8_t duty)
{
    uint32_t period = 4800;
    uint32_t pulse_width = (period * duty) / 100;
    
    PWMPulseWidthSet(PWM0_BASE, PWM_OUT_7, pulse_width);
    PWMOutputState(PWM0_BASE, PWM_OUT_7_BIT, true);

    GPIOPinWrite(GPIO_PORTE_BASE, 0x03, 0x02); 
    
    UARTprintf("Motor activado al %d%%\n", duty);
}

void motor_stop()
{
    PWMOutputState(PWM0_BASE, PWM_OUT_7_BIT, false);
    
    GPIOPinWrite(GPIO_PORTE_BASE, 0x03, 0x00); 
    
    UARTprintf("Motor detenido\n");
}