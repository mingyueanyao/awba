################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Sources/benchm.cpp 

OBJS += \
./Sources/benchm.o 

CPP_DEPS += \
./Sources/benchm.d 


# Each subdirectory must supply rules for building sources it contributes
Sources/%.o: ../Sources/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I"e:\WinApps\mingw64\x86_64-w64-mingw32\include" -I"e:\WinApps\mingw64\include" -I"e:\WinApps\mingw64\x86_64-w64-mingw32\include\c++\" -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


