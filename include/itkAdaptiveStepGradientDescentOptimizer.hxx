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

#include "itkAdaptiveStepGradientDescentOptimizer.h"

namespace itk
{
/**
 * Constructor
 */
AdaptiveStepGradientDescentOptimizer
::AdaptiveStepGradientDescentOptimizer() : Superclass(),
    m_Fmax(1),
    m_Fmin(-0.5),
    m_omega(1),
    m_CurrentTime(0),
    m_CurrentStep(0),
    m_BestCurrentTime(0),
    m_Previous_a(this->m_a),
    m_OrthantProjection(false),
    m_Stop(false),
    m_Value(0),
    m_BestValue(std::numeric_limits<MeasureType>::max()),
    m_StopCondition(Unknown),
    m_CurrentIteration(0)
{

    itkDebugMacro("Constructor");
    m_StopConditionDescription.str("");
 
    m_PreviousGradient.Fill(0.0f);
    m_BestPosition.Fill(0.0f);
}

void
AdaptiveStepGradientDescentOptimizer
::StartOptimization(void)
{
    itkDebugMacro("StartOptimization");

    m_CurrentTime   = 0;
    m_BestCurrentTime = 0;

    m_StopCondition = Unknown;
    m_StopConditionDescription.str("");
    m_StopConditionDescription << this->GetNameOfClass() << ": ";

    const unsigned int spaceDimension = m_CostFunction->GetNumberOfParameters();
    m_Gradient = DerivativeType(spaceDimension);
    m_BestGradient = DerivativeType(spaceDimension);
    m_PreviousGradient = DerivativeType(spaceDimension);
    m_BestPreviousGradient = DerivativeType(spaceDimension);
    m_Gradient.Fill(0.0f);
    m_BestGradient.Fill(0.0f);
    m_PreviousGradient.Fill(0.0f);
    m_BestPreviousGradient.Fill(0.0f);

    m_CurrentStep = 0;
    m_Previous_a = this->m_a;

    if(this->m_Maximize){
        m_BestValue = std::numeric_limits<MeasureType>::lowest();
    }
    else{
        m_BestValue = std::numeric_limits<MeasureType>::max();
    }

    this->SetCurrentPosition( GetInitialPosition() );

    m_BestPosition = this->GetCurrentPosition();

    this->ResumeOptimization();
}


/**
 * Resume the optimization
 */
void
AdaptiveStepGradientDescentOptimizer
::ResumeOptimization(void)
{
    itkDebugMacro("ResumeOptimization");

    m_Stop = false;

    this->InvokeEvent( StartEvent() );

    while ( !m_Stop ){
        if ( m_CurrentIteration >= this->GetNumberOfIterations()){
            m_StopCondition = MaximumNumberOfIterations;
            m_StopConditionDescription << "Maximum number of iterations ("
                                       << this->GetNumberOfIterations()
                                       << ") exceeded.";
            this->StopOptimization();
            break;
        }

        m_PreviousGradient = m_Gradient;

        try{
            m_CostFunction->GetValueAndDerivative(
                        this->GetCurrentPosition(), m_Value, m_Gradient);
        }
        catch ( ExceptionObject & excp ){
            m_StopCondition = CostFunctionError;
            m_StopConditionDescription << "Cost function error after "
                                       << m_CurrentIteration
                                       << " iterations. "
                                       << excp.GetDescription();
            this->StopOptimization();
            throw excp;
        }

        // new value is better than best value
        if((this->m_Maximize && m_Value>m_BestValue) ||
                (!this->m_Maximize && m_Value<m_BestValue)){
            m_BestValue = m_Value;
            m_BestPreviousGradient = m_PreviousGradient;
            m_BestGradient = m_Gradient;
            m_BestPosition = this->GetCurrentPosition();
            m_BestCurrentTime = m_CurrentTime;

            this->AdvanceOneStep();
        }
        // new value is worse than best balue
        else{

            this->AdvanceOneStep();
        }

        if ( m_Stop ){
            break;
        }

        if(m_CurrentStep < 1e-6){
             m_StopCondition = StepTooSmall;
             m_StopConditionDescription << "a have become too small. Optimization converged after "
                                           << this->GetNumberOfIterations() << " iterations.";
             this->StopOptimization();
             break;
        }


        m_CurrentIteration++;
    }

    m_StopCondition = MaximumNumberOfIterations;
    m_StopConditionDescription << "Maximum number of iterations has been reached: "
                                  << this->GetNumberOfIterations() << " iterations.";

    //this->SetCurrentPosition( m_BestPosition );
}

/**
 * Stop optimization
 */
void
AdaptiveStepGradientDescentOptimizer
::StopOptimization(void)
{
  itkDebugMacro("StopOptimization");

  m_Stop = true;
  this->InvokeEvent( EndEvent() );
}

void
AdaptiveStepGradientDescentOptimizer
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

/**
 * Advance one Step following the gradient direction
 */
void
AdaptiveStepGradientDescentOptimizer
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
    DerivativeType previousTransformedGradient(spaceDimension);

    for ( unsigned int j = 0; j < spaceDimension; j++ ){
        transformedGradient[j] = m_Gradient[j] / scales[j];
        previousTransformedGradient[j] = m_PreviousGradient[j] / scales[j];
    }

    /// Step along current gradient with iteration dependent step size
    auto gamma = [this](double t){ return this->m_a/std::pow(this->m_A+t+1,this->m_alpha);};
    m_CurrentStep = gamma(m_CurrentTime);

    ParametersType newPosition(spaceDimension);
    for ( unsigned int j = 0; j < spaceDimension; j++ ){
        newPosition[j] = currentPosition[j]
                + direction * m_CurrentStep * transformedGradient[j];
    }

    std::cout << "\t a: " << this->m_a << std::flush;
    std::cout << "\t time: " << m_CurrentTime << std::flush;
    std::cout << "\t step: " << m_CurrentStep << std::flush;

    /// orthant projection
    if(m_OrthantProjection){
        for ( unsigned int j = 0; j < spaceDimension; j++ ){
            if(newPosition[j]*currentPosition[j] < 0){
                newPosition[j] = 0;
            }
        }
    }

    /// Calculate next time step
    // calculate scalar product
    double scalarProduct = 0;
    for ( unsigned int i = 0; i < spaceDimension; i++ ){
        const double weight1 = transformedGradient[i];
        const double weight2 = previousTransformedGradient[i];
        scalarProduct += weight1 * weight2;
    }
    std::cout << "\t scalar product: " << scalarProduct << std::flush;

    // update time
    auto f = [this](double x) {
        return m_Fmin + (m_Fmax-m_Fmin)/(1 - (m_Fmax/(m_Fmin))*std::exp(-x/m_omega) );
    };
    m_CurrentTime = std::max(0.0, m_CurrentTime+f(-scalarProduct));

    /// Calculate gradient magnitude, just for info
    double magnitudeSquare = 0;
    for ( unsigned int dim = 0; dim < spaceDimension; dim++ ){
      const double weighted = transformedGradient[dim];
      magnitudeSquare += weighted * weighted;
      }
    const double gradientMagnitude = std::sqrt(magnitudeSquare);
    std::cout << "\t magnitude: " << gradientMagnitude << std::flush;

    /// print out number of zero parameters
    unsigned num_zeros = 0;
    for(unsigned p=0; p<newPosition.Size(); p++){
        if(newPosition.GetElement(p)==0) num_zeros++;
    }
    std::cout << "\t zero params: " << 100.0*static_cast<double>(num_zeros)/static_cast<double>(newPosition.Size()) << "%" << std::flush;

    this->SetCurrentPosition(newPosition);

    this->InvokeEvent( IterationEvent() );
}
} // end namespace itk

