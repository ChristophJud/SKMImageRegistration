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

#include "itkStandardGradientDescentOptimizerv4.h"

namespace itk
{

template<typename TInternalComputationValueType>
StandardGradientDescentOptimizerv4<TInternalComputationValueType>
::StandardGradientDescentOptimizerv4() : Superclass(),
    m_a(1),
    m_A(5),
    m_alpha(1),
    m_OrthantProjection(false),
    m_CalculateMagnitude(false),
    m_CountZeroParams(false)
{
    // set superclass stuff
    this->DoEstimateScalesOff();
    this->DoEstimateLearningRateAtEachIterationOn();
}

template<typename TInternalComputationValueType>
void
StandardGradientDescentOptimizerv4<TInternalComputationValueType>
::PrintSelf(std::ostream & os, Indent indent) const
{
    Superclass::PrintSelf(os, indent);

    os << indent << "a: "
       << this->m_a << std::endl;
    os << indent << "A: "
       << this->m_A << std::endl;
    os << indent << "alpha: "
       << this->m_alpha << std::endl;
    if(m_OrthantProjection)
        os << indent << "orthant projection: On";
    else
        os << indent << "orthant projection: Off";
    os << std::endl;
}

template<typename TInternalComputationValueType>
void
StandardGradientDescentOptimizerv4<TInternalComputationValueType>
::ModifyGradientByLearningRateOverSubRange( const IndexRangeType& subrange )
{
    /* Loop over the range. It is inclusive. */
    for ( IndexValueType j = subrange[0]; j <= subrange[1]; j++ ){
        // orthant projection
        if(m_OrthantProjection &&
                this->m_Gradient.Size() == this->m_PreviousGradient.Size() &&
                this->m_Gradient[j]*this->m_PreviousGradient[j] < 0){
            this->m_Gradient[j] = 0;
        }
        else{
            this->m_Gradient[j] = this->m_Gradient[j] * this->m_LearningRate;
        }
    }

    if(m_CalculateMagnitude){
    /// Calculate gradient magnitude, just for info
        TInternalComputationValueType magnitudeSquare = 0;
        for ( IndexValueType j = subrange[0]; j <= subrange[1]; j++ ){
            const auto weighted = this->m_Gradient[j];
            magnitudeSquare += weighted * weighted;
        }
        const TInternalComputationValueType gradientMagnitude = std::sqrt(magnitudeSquare);
        std::cout << "\t magnitude: " << gradientMagnitude << std::flush;
    }

    if(m_CountZeroParams){
        unsigned num_zeros = 0;
        for ( IndexValueType j = subrange[0]; j <= subrange[1]; j++ ){
            if(this->m_Gradient[j]==0) 
                num_zeros++;
        }
        std::cout << "\t zero params: " 
                  << 100.0*static_cast<TInternalComputationValueType>(num_zeros) /
                     static_cast<TInternalComputationValueType>(this->m_Gradient.Size()) 
                  << "%" << std::flush;
    }
}

template<typename TInternalComputationValueType>
void
StandardGradientDescentOptimizerv4<TInternalComputationValueType>
::EstimateLearningRate()
{
        this->m_LearningRate = std::pow(this->m_a / 
                                        (this->m_A+this->GetCurrentIteration()+1), this->m_alpha);
}


}