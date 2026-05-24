/***************************************************************************//**
* @author  Ben Haubrich
* @file    LcdConfiguration.hpp
* @details Lcd configuration
*******************************************************************************/
#ifndef __LCD_CONFIGURATION_HPP__
#define __LCD_CONFIGURATION_HPP__

//AbstractionLayer
#include "LcdFactory.hpp"

namespace Rvt35AHBNWC00 {

    inline RiverdiEve3Tft35InchTypes::Configuration LcdConfiguration() {
        constexpr SpiTypes::SpiParams::HardwareConfig spiHardwareConfig = {
            .peripheral = APP_LCD_SPI_PERIPHERAL_NUMBER,
            .periperhalOutControllerIn = APP_LCD_POCI_PIN_NUMBER,
            .perpheralInControllerOut = APP_LCD_PICO_PIN_NUMBER,
            .chipSelect = APP_LCD_CHIP_SELECT_PIN_NUMBER,
            .clock = APP_LCD_CLOCK_PIN_NUMBER,
            .chipSelectMode = APP_LCD_SPI_CHIPSLECT_MODE,
            .chipSelectGpioPeripheral = APP_LCD_SPI_CHIP_SELECT_GPIO_PERIPHERAL_NUMBER,
            .chipSelectGpioPin = APP_LCD_SPI_CHIP_SELECT_GPIO_PIN_NUMBER
        };
        constexpr SpiTypes::SpiParams::DriverConfig spiDriverConfig = {
            .isController = true,
            .chipSelectActiveLow = true,
            .format = SpiTypes::FrameFormat::Mode0,
            .clockFrequency = Hertz(APP_LCD_SPI_CLOCK_FREQUENCY),
            .dataSize = SpiTypes::DataSize::EightBits,
            .channels = SpiTypes::Channels::Single
        };
        constexpr SpiTypes::SpiParams spiParams(spiHardwareConfig, spiDriverConfig);

        constexpr GpioTypes::GpioParams::HardwareConfig gpioHardwareConfig = {
            .peripheralNumber = APP_LCD_POWERDOWN_PERIPHERAL_NUMBER,
            .pinNumber = APP_LCD_POWERDOWN_PIN_NUMBER,
            .driveType = GpioTypes::DriveType::PushPull,
            .driveStrength = GpioTypes::DriveStrength::EightMilliAmps
        };
        constexpr GpioTypes::GpioParams::InterruptConfig gpioInterruptConfig = {
            .interruptFlags = GpioTypes::Interrupts::Disabled,
            .interruptCallback = nullptr
        };

        constexpr GpioTypes::GpioParams gpioParams(gpioHardwareConfig, gpioInterruptConfig);

        return RiverdiEve3Tft35InchTypes::Configuration(spiParams, gpioParams);
    }
}

#endif // __LCD_CONFIGURATION_HPP__