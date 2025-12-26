/***************************************************************************//**
* @author   Ben Haubrich
* @file     main.cpp
* @date     Wednesday, February 26th, 2025
*******************************************************************************/
//AbstractionLayer
#include "PowerResetClockManagementModule.hpp"
#include "GpioModule.hpp"
#include "Log.hpp"
//EdgeInference
#include "EdgeInference.hpp"

static void startEdgeInference(EdgeInference &edgeInference) {
    Id edgeInferenceId;

    OperatingSystem::Instance().createThread(OperatingSystemTypes::Priority::Normal,
                                             EdgeInference::_EdgeInferenceThreadName,
                                             &edgeInference,
                                             APP_DEFAULT_STACK_SIZE,
                                             startEdgeInferenceThread,
                                             edgeInferenceId);

    OperatingSystem::Instance().startScheduler();

    //Platforms (especially desktop platforms like Linux and Darwin) that don't have the concept of starting a scheduler will
    //join threads instead. Other targets that run on embedded RTOSs like Azure, Zephyr, and FreeRTOS never return after
    //starting the scheduler.
    assert(ErrorType::NoData != OperatingSystem::Instance().joinThread(EdgeInference::_EdgeInferenceThreadName));
}

static void initGlobals() {
    OperatingSystem::Init();
    Logger::Init();
    EdgeInference::Init();
}

#if __XTENSA__
extern "C" void app_main() {
#else
int main(void) {
#endif
    PowerResetClockManagement prcm;
    Gpio gpio;

    constexpr auto isNonCriticalError = [](const ErrorType error) -> bool {
        return (ErrorType::NotAvailable == error || ErrorType::NotSupported == error || ErrorType::NotImplemented == error);
    };

    ErrorType error = prcm.init();

    if (ErrorType::Success == error || isNonCriticalError(error)) {
        error = prcm.setClockFrequency(Hertz(APP_PROCESSOR_CLOCK_FREQUENCY), Hertz(APP_EXTERNAL_CRYSTAL_FREQUENCY));

        if (ErrorType::Success == error || isNonCriticalError(error)) {
            error = gpio.configure(GpioTypes::GpioParams());

            if (ErrorType::Success == error || isNonCriticalError(error)) {
                error = gpio.init();

                initGlobals();
                OperatingSystem::Instance().setTimeOfDay(UnixTime(0), Seconds(0));
                startEdgeInference(EdgeInference::Instance());
            }
        }
    }

#if __XTENSA__
    return;
#else
    return 0;
#endif
}
