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

    while(1)
    {
        UARTgets(data, 100);

        char nivel = data[0];  
        UARTprintf("Nivel recibido: %c\n", nivel);

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
    }
}

void setup_gpio() {

    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOB);
    while(!SysCtlPeripheralReady(SYSCTL_PERIPH_GPIOB))
    {
    }

    GPIOPinTypeGPIOOutput(GPIO_PORTB_BASE, 0x30);
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