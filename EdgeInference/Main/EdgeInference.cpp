#include "EdgeInference.hpp"
#include "quantizedHandwrittenZeroToNineModel.hpp"
//AbstractionLayer
#include "Inference.hpp"

ErrorType EdgeInference::edgeInferenceThread() {
    MachineLearningInference inference;
    ErrorType error = LcdFactory::Factory<APP_LCD_PART_NUMBER>(_lcd);

    if (ErrorType::Success == (error = inference.init(1))) {
        const char *modelData = reinterpret_cast<const char *>(TfLiteModels::quantizedHandwrittenZeroToNineModel.data());
        const size_t modelSize = TfLiteModels::quantizedHandwrittenZeroToNineModel.size();

        if (ErrorType::Success == (error = inference.loadModel(std::string_view(modelData, modelSize)))) {
            error = std::visit([&](auto &lcd) -> ErrorType {
                ErrorType error = ErrorType::Failure;

                if (ErrorType::Success == (error = lcd.configure())) {
                    error = lcd.init();

                    if (ErrorType::Success == error) {
                        error = lcd.startDesign();

                        if (ErrorType::Success == error) {
                            LcdTypes::FreehandSketch sketch;
                            sketch.area = {.origin = {0,0}, .width = lcd.screenParameters().activeArea.width/2, .height = lcd.screenParameters().activeArea.height/2};
                            //Background colour is black and the brush colour is white to match the dataset.
                            sketch.paperColour = 0x000000;
                            sketch.brushColour = 0xFFFFFF;
                            error = lcd.addDesignElement(sketch);

                            LcdTypes::Button button;
                            button.area = {.origin = {lcd.screenParameters().activeArea.width,
                                                      lcd.screenParameters().activeArea.height},
                                           .width = lcd.screenParameters().activeArea.width/8,
                                           .height = lcd.screenParameters().activeArea.height/8};
                            button.area.origin.x -= button.area.width;
                            button.area.origin.y -= button.area.height;
                            button.font = 26;
                            button.id = 2;
                            button.text = StaticString::Data<sizeof("Enter")>("Enter");
                            error = lcd.addDesignElement(button);
                            lcd.endDesign();

                            while (ErrorType::Negative == lcd.waitForTouches({button.id}, Milliseconds(UINT32_MAX)));

                            std::string screenBuffer(sketch.area.size(), 0);
                            screenBuffer.resize(sketch.area.size());
                            error = lcd.copyScreen(screenBuffer, sketch.area, LcdTypes::PixelFormat::Greyscale);

                            if (ErrorType::Success == error) {
                                error = inference.setInput(std::string_view(screenBuffer.data(), sketch.area.size()), 0);

                                if (ErrorType::Success == error) {
                                    error = inference.runInference();

                                    if (ErrorType::Success == error) {
                                        error = inference.getOutput(screenBuffer, 0);

                                        int i = 0;
                                        for (int8_t outputScore : screenBuffer) {
                                            PLT_LOGI(EdgeInference::TAG, "Probability of %d %d", i, static_cast<int8_t>(outputScore));
                                            i++;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                return error;

            }, *_lcd);
        }
    }

    return error;
}

#ifdef __cplusplus
extern "C" {
#endif
void *startEdgeInferenceThread(void *arg) {
    ErrorType error = static_cast<EdgeInference *>(arg)->edgeInferenceThread();
    PLT_LOGW(EdgeInference::TAG, "EdgeInference thread exited with error %u", (uint8_t)error);
    return nullptr;
}
#ifdef __cplusplus
}
#endif
