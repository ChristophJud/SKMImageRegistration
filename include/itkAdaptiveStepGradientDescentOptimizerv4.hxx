/*
 * Copyright 2016 University of Basel, Medical Image Analysis Center
 *
 * Author: Christoph Jud     (christoph.jud@unibas.ch)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 
#pragma once

#include "itkAdaptiveStepGradientDescentOptimizerv4.h"

namespace itk
{

template<typename TInternalComputationValueType>
AdaptiveStepGradientDescentOptimizerv4<TInternalComputationValueType>
::AdaptiveStepGradientDescentOptimizerv4() : Superclass(),
    m_Fmax(1),
    m_Fmin(-0.5),
    m_omega(1),
    m_CurrentTime(0)
{
    this->m_UseConvergenceMonitoring = false;
}

template<typename TInternalComputationValueType>
void
AdaptiveStepGradientDescentOptimizerv4<TInternalComputationValueType>
::PrintSelf(std::ostream & os, Indent indent) const
{
    Superclass::PrintSelf(os, indent);

    os << indent << "F_max: "
       << m_Fmax << std::endl;
    os << indent << "F_min: "
       << m_Fmin << std::endl;
    os << indent << "omega: "
       << m_omega << std::endl;
}

template<typename TInternalComputationValueType>
void
AdaptiveStepGradientDescentOptimizerv4<TInternalComputationValueType>
::StartOptimization( bool doOnlyInitialization )
{
    /* Must call the superclass version for basic validation and setup */
    Superclass::StartOptimization( doOnlyInitialization );

    m_CurrentTime = 0;
}

template<typename TInternalComputationValueType>
void
AdaptiveStepGradientDescentOptimizerv4<TInternalComputationValueType>
::ResumeOptimization()
{
    this->m_StopConditionDescription.str("");
    this->m_StopConditionDescription << this->GetNameOfClass() << ": ";
    this->InvokeEvent( StartEvent() );

    this->m_Stop = false;
    while( ! this->m_Stop ){
        // Do not run the loop if the maximum number of iterations is reached or its value is zero.
        if ( this->m_CurrentIteration >= this->m_NumberOfIterations ){
            this->m_StopConditionDescription << "Maximum number of iterations (" << this->m_NumberOfIterations << ") exceeded.";
            this->m_StopCondition = Superclass::MAXIMUM_NUMBER_OF_ITERATIONS;
            this->StopOptimization();
            break;
        }

        // Save previous value with shallow swap that will be used by child optimizer.
        swap( this->m_PreviousGradient, m_TemporalGradient );

        /* Compute metric value/derivative. */
        try{
            /* m_Gradient will be sized as needed by metric. If it's already
            * proper size, no new allocation is done. */
            this->m_Metric->GetValueAndDerivative( this->m_CurrentMetricValue, this->m_Gradient );
        }
        catch ( ExceptionObject & err ){
        this->m_StopCondition = Superclass::COSTFUNCTION_ERROR;
        this->m_StopConditionDescription << "Metric error during optimization";
        this->StopOptimization();

            // Pass exception to caller
        throw err;
        }

        m_TemporalGradient = this->m_Gradient;

        /* Check if optimization has been stopped externally.
        * (Presumably this could happen from a multi-threaded client app?) */
        if ( this->m_Stop ){
            this->m_StopConditionDescription << "StopOptimization() called";
            break;
        }

        /* Check the convergence by WindowConvergenceMonitoringFunction.
        */
        if ( this->m_UseConvergenceMonitoring ){
            this->m_ConvergenceMonitoring->AddEnergyValue( this->m_CurrentMetricValue );
            try{
                this->m_ConvergenceValue = this->m_ConvergenceMonitoring->GetConvergenceValue();
                if (this->m_ConvergenceValue <= this->m_MinimumConvergenceValue){
                    this->m_StopConditionDescription << "Convergence checker passed at iteration " << this->m_CurrentIteration << ".";
                    this->m_StopCondition = Superclass::CONVERGENCE_CHECKER_PASSED;
                    this->StopOptimization();
                    break;
                }
            }
            catch(std::exception & e){
                std::cerr << "GetConvergenceValue() failed with exception: " << e.what() << std::endl;
            }
        }

        /* Advance one step along the gradient.
        * This will modify the gradient and update the transform. */
        this->AdvanceOneStep();

        /* Store best value and position */
        if ( this->m_ReturnBestParametersAndValue && this->m_CurrentMetricValue < this->m_CurrentBestValue ){
            this->m_CurrentBestValue = this->m_CurrentMetricValue;
            this->m_BestParameters = this->GetCurrentPosition( );
        }

        /* Update and check iteration count */
        this->m_CurrentIteration++;

    } //while (!m_Stop)
}

template<typename TInternalComputationValueType>
void
AdaptiveStepGradientDescentOptimizerv4<TInternalComputationValueType>
::ModifyGradientByLearningRateOverSubRange( const IndexRangeType& subrange )
{
    auto scalarproduct = this->CalculateScalarProduct(subrange);

    Superclass::ModifyGradientByLearningRateOverSubRange(subrange);

    this->ModifyCurrentTime(scalarproduct);
}

template<typename TInternalComputationValueType>
void
AdaptiveStepGradientDescentOptimizerv4<TInternalComputationValueType>
::EstimateLearningRate()
{
    auto gamma = [this](double t){ return this->m_a/std::pow(this->m_A+t+1,this->m_alpha);};
    this->m_LearningRate = gamma(m_CurrentTime);

    std::cout << "\t a: " << this->m_a << std::flush;
    std::cout << "\t time: " << m_CurrentTime << std::flush;
    std::cout << "\t step: " << this->m_LearningRate << std::flush;
}


template<typename TInternalComputationValueType>
typename AdaptiveStepGradientDescentOptimizerv4<TInternalComputationValueType>
::InternalComputationValueType AdaptiveStepGradientDescentOptimizerv4<TInternalComputationValueType>
::CalculateScalarProduct(const IndexRangeType& subrange )
{
    if(this->m_Gradient.Size() != this->m_PreviousGradient.Size())
        this->m_PreviousGradient = this->m_Gradient;

    /// Calculate next time step
    // calculate scalar product
    TInternalComputationValueType scalarProduct = 0;
    for ( IndexValueType j = subrange[0]; j <= subrange[1]; j++ ){
        scalarProduct += this->m_Gradient[j] * this->m_PreviousGradient[j];
    }
    std::cout << "\t scalar product: " << scalarProduct << std::flush;

    return scalarProduct;
}

template<typename TInternalComputationValueType>
void
AdaptiveStepGradientDescentOptimizerv4<TInternalComputationValueType>
::ModifyCurrentTime(InternalComputationValueType scalarProduct )
{
    // update time
    auto f = [this](double x) {
        return m_Fmin + (m_Fmax-m_Fmin)/(1 - (m_Fmax/(m_Fmin))*std::exp(-x/m_omega) );
    };
    m_CurrentTime = std::max(0.0, m_CurrentTime+f(-scalarProduct));
}

}