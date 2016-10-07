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
    this->m_LearningRate = 0;
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
    if( this->m_Metric.IsNull() ){
        itkExceptionMacro("m_Metric must be set.");
        return;
    }

    m_CurrentTime = 0;
    m_TemporaryGradient = DerivativeType(this->m_Metric->GetNumberOfParameters());
    m_TemporaryGradient.Fill(0.0);

    /* Must call the superclass version for basic validation and setup */
    Superclass::StartOptimization( doOnlyInitialization );
}

template<typename TInternalComputationValueType>
void
AdaptiveStepGradientDescentOptimizerv4<TInternalComputationValueType>
::ModifyGradientByLearningRateOverSubRange( const IndexRangeType& subrange )
{
    auto scalarproduct = CalculateScalarProduct(subrange);

    // store current gradient
    m_TemporaryGradient = this->m_Gradient;

    Superclass::ModifyGradientByLearningRateOverSubRange(subrange);

    ModifyCurrentTime(scalarproduct);
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
::CalculateScalarProduct(const IndexRangeType& subrange ) const
{
    // calculate scalar product of 
    // previous gradient (m_TemporaryGraident) 
    // and current gradient (m_Gradient)
    TInternalComputationValueType scalarProduct = 0;
    #pragma omp parallel for reduction (+:scalarProduct)
    for ( IndexValueType j = subrange[0]; j <= subrange[1]; j++ ){
        scalarProduct += this->m_Gradient[j] * m_TemporaryGradient[j];
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