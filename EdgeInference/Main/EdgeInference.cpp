#include "EdgeInference.hpp"
#include "quantizedHandwrittenZeroToNineModel.hpp"
//AbstractionLayer
#include "Inference.hpp"
//C++
#include <cinttypes>

ErrorType EdgeInference::edgeInferenceThread() {
    MachineLearningInference inference;
    ErrorType error = LcdFactory::Factory<APP_RIVERDI_LCD_PART_NUMBER>(_lcd);

    if (ErrorType::Success == (error = inference.init())) {
        const char *modelData = reinterpret_cast<const char *>(TfLiteModels::quantizedHandwrittenZeroToNineModel.data());
        const size_t modelSize = TfLiteModels::quantizedHandwrittenZeroToNineModel.size();

        if (ErrorType::Success == (error = inference.loadModel(std::string_view(modelData, modelSize)))) {
            error = std::visit([&](auto &lcd) -> ErrorType {
                ErrorType error = ErrorType::Failure;

                if (ErrorType::Success == (error = lcd.configure())) {
                    error = lcd.init();

                    while (ErrorType::Success == error) {
                        error = lcd.startDesign();

                        if (ErrorType::Success == error) {
                            LcdTypes::FreehandSketch sketch;
                            //Not a typo. We want the sketch area to be square.
                            const uint32_t sketchWidth = lcd.screenParameters().activeArea.height / 2;
                            sketch.area = {.origin = {0,0}, .width = sketchWidth, .height = sketchWidth};
                            //Background colour is black and the brush colour is white to match the dataset.
                            sketch.paperColour = 0x000000;
                            sketch.brushColour = 0xFFFFFF;
                            error = lcd.addDesignElement(sketch);

                            if (ErrorType::Success == error) {
                                LcdTypes::Button button;
                                button.area = {.origin = {lcd.screenParameters().activeArea.width,
                                                        lcd.screenParameters().activeArea.height},
                                            .width = lcd.screenParameters().activeArea.width/8,
                                            .height = lcd.screenParameters().activeArea.height/8};
                                button.area.origin.x -= button.area.width;
                                button.area.origin.y -= button.area.height;
                                button.font = 26;
                                button.id = 2;
                                button.text = StaticString::Container("Enter");
                                error = lcd.addDesignElement(button);
                                error = lcd.endDesign();

                                while (ErrorType::Negative == lcd.waitForTouches({button.id}, Milliseconds(UINT32_MAX)));

                                //TODO: Could be StaticString now that the sketch is smaller.
                                std::string screenBuffer(sketch.area.size(), 0);
                                error = lcd.copyScreen(screenBuffer, sketch.area, PixelFormat::Greyscale);

                                if (ErrorType::Success == error) {
                                    error = inferencePreprocessing(screenBuffer, sketch.area);

                                    if (ErrorType::Success == error) {
                                        error = inference.setInput(std::string_view(screenBuffer.data(), screenBuffer.size()), 0);

                                        if (ErrorType::Success == error) {
                                            error = inference.runInference();

                                            if (ErrorType::Success == error) {
                                                error = inference.getOutput(screenBuffer, 0);
                                                uint8_t inferredDigit = 0;
                                                int8_t inferredDigitConfidence = -128;

                                                for (size_t i = 0; i < screenBuffer.size(); i++) {

                                                    if (inferredDigitConfidence < static_cast<int8_t>(screenBuffer[i])) {
                                                        inferredDigitConfidence = screenBuffer[i];
                                                        inferredDigit = i;
                                                    }
                                                }

                                                lcd.startDesign();
                                                LcdTypes::Text resultText;
                                                resultText.location = {lcd.screenParameters().activeArea.width/2, lcd.screenParameters().activeArea.height/2};
                                                resultText.colour = 0x00FFFFFF;
                                                resultText.font = 31;
                                                resultText.text = StaticString::Container("9 (100.0%)");
                                                const Percent confidence = (static_cast<float>(inferredDigitConfidence + 128) / UINT8_MAX) * 100.0f;
                                                const Bytes written = snprintf(resultText.text->data(), resultText.text->capacity(), "%u (%" PRIu32 "%%)", inferredDigit, static_cast<uint32_t>(confidence));
                                                resultText.text->resize(written);
                                                error = lcd.addDesignElement(resultText);
                                                lcd.endDesign();
                                                OperatingSystem::Instance().delay(Milliseconds(2000));
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        error = lcd.clearScreen(0x0000FF);
                    }
                }

                return error;

            }, *_lcd);
        }
    }

    return error;
}

ErrorType EdgeInference::inferencePreprocessing(std::string &screenBuffer, const Area &area) {
    ErrorType error = Binarize(screenBuffer, PixelFormat::Greyscale);

    if (ErrorType::Success == error) {
        const Area islandArea = {.origin = {0,0}, .width = 3, .height = 3};
        error = IslandFilter(screenBuffer, area, 0xFF, 0x00, islandArea);

        if (ErrorType::Success == error) {
            Area downsized = {{0,0},28,28};
            error = DownsizeImage(area, downsized, ImageResampling::Box, PixelFormat::Greyscale, screenBuffer);

            if (ErrorType::Success == error) {

                error = Binarize(screenBuffer, PixelFormat::Greyscale);

                if (ErrorType::Success == error) {
                    std::string dilated(screenBuffer.size(), 0);
                    error = Dilate(screenBuffer, downsized, {.origin = {0,0}, .width = 2, .height = 2}, PixelFormat::Greyscale, 0xFF, 0xFF, dilated);

                    screenBuffer = std::move(dilated);
                }
            }
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
