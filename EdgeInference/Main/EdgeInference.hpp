/***************************************************************************//**
* @author  Ben Haubrich
* @file    EdgeInference.hpp
* @details Application code for Edge Inference 
*******************************************************************************/
#ifndef __SM10001_SLIDE_POTENTIOMETER_HPP__
#define __SM10001_SLIDE_POTENTIOMETER_HPP__

//AbstractionLayer
#include "Global.hpp"
#include "OperatingSystemModule.hpp"
#include "LcdFactory.hpp"

class EdgeInference : public Global<EdgeInference> {

    public:
    EdgeInference() : Global<EdgeInference>() {}

    static constexpr char TAG[] = "EdgeInference";

    static constexpr std::array<char, OperatingSystemTypes::MaxThreadNameLength> _EdgeInferenceThreadName = {"edgeInference"};

    ErrorType edgeInferenceThread();

    private:
    std::optional<LcdFactoryTypes::LcdFactoryVariant> _lcd;
};

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief Creates the slide potentiometer code.
*/
void *startEdgeInferenceThread(void *arg);
#ifdef __cplusplus
}
#endif
#endif // __SM10001_SLIDE_POTENTIOMETER_HPP__