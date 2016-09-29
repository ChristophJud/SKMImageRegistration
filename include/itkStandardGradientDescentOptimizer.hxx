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

#include "itkStandardGradientDescentOptimizer.h"

namespace itk
{
/**
 * Constructor
 */
StandardGradientDescentOptimizer
::StandardGradientDescentOptimizer() : Superclass(),
    m_a(1),
    m_A(5),
    m_alpha(1)
{
    itkDebugMacro("Constructor");
}

void
StandardGradientDescentOptimizer
::PrintSelf(std::ostream & os, Indent indent) const
{
    Superclass::PrintSelf(os, indent);

    os << indent << "a: "
       << m_a << std::endl;
    os << indent << "A: "
       << m_A << std::endl;
    os << indent << "alpha: "
       << m_alpha << std::endl;
}

/**
 * Advance one Step following the gradient direction
 */
void
StandardGradientDescentOptimizer
::AdvanceOneStep(void)
{
    itkDebugMacro("AdvanceOneStep");

    double direction;
    if ( this->m_Maximize ){
        direction = 1.0;
    }
    else{
        direction = -1.0;
    }

    const unsigned int spaceDimension =  m_CostFunction->GetNumberOfParameters();
    const ParametersType & currentPosition = this->GetCurrentPosition();
    const ScalesType & scales = this->GetScales();

    // Make sure the scales have been set properly
    if ( scales.size() != spaceDimension ){
        itkExceptionMacro(<< "The size of Scales is "
                          << scales.size()
                          << ", but the NumberOfParameters for the CostFunction is "
                          << spaceDimension
                          << ".");
    }

    DerivativeType transformedGradient(spaceDimension);

    for ( unsigned int j = 0; j < spaceDimension; j++ ){
        transformedGradient[j] = m_Gradient[j] / scales[j];
    }

    /// iteration dependent step size
    double factor = std::pow(m_a/(m_A+this->GetCurrentIteration()+1),m_alpha);

    std::cout << "\t step: " << factor << std::flush;

    ParametersType newPosition(spaceDimension);
    for ( unsigned int j = 0; j < spaceDimension; j++ ){
        newPosition[j] = currentPosition[j]
                + direction * factor * transformedGradient[j];
    }

    this->SetCurrentPosition(newPosition);

    this->InvokeEvent( IterationEvent() );
}
} // end namespace itk

