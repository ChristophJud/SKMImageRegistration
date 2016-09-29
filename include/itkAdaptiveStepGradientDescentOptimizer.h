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

#include "itkIntTypes.h"
#include "itkStandardGradientDescentOptimizer.h"
#include <string>

namespace itk
{
/** \class StandardGradientDescentOptimizer
 *
 * \ingroup Numerics Optimizers
 * \ingroup ITKOptimizers
 */
class AdaptiveStepGradientDescentOptimizer:
  public StandardGradientDescentOptimizer
{
public:
  /** Standard class typedefs. */
  typedef AdaptiveStepGradientDescentOptimizer  Self;
  typedef StandardGradientDescentOptimizer              Superclass;
  typedef SmartPointer< Self >                  Pointer;
  typedef SmartPointer< const Self >            ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(AdaptiveStepGradientDescentOptimizer, StandardGradientDescentOptimizer);


  /** Codes of stopping conditions. */
  typedef enum {
    GradientMagnitudeTolerance = 1,
    StepTooSmall = 2,
    ImageNotAvailable = 3,
    CostFunctionError = 4,
    MaximumNumberOfIterations = 5,
    Unknown = 6
    } StopConditionType;

  /** Advance one step following the gradient direction. */
  virtual void AdvanceOneStep();

  /** Start optimization. */
  virtual void    StartOptimization(void) ITK_OVERRIDE;

  /** Resume previously stopped optimization with current parameters.
   * \sa StopOptimization */
  void    ResumeOptimization();

  /** Stop optimization.
   * \sa ResumeOptimization */
  void    StopOptimization();

  itkGetConstMacro(Fmax, double);
  void SetFmax(double fmax){
      if(fmax>0){
          m_Fmax = fmax;
      }
      else{
          itkExceptionMacro("F_max has to be positive");
      }
  }

  itkGetConstMacro(Fmin, double);
  void SetFmin(double fmin){
      if(fmin<0){
          m_Fmin = fmin;
      }
      else{
          itkExceptionMacro("F_min ha to be negative")
      }
  }

  itkGetConstMacro(omega, double);
  void Setomega(double omega){
      if(omega>0){
          m_omega = omega;
      }
      else{
          itkExceptionMacro("omega has to be positive.")
      }
  }

  itkGetConstMacro(OrthantProjection,bool);
  itkSetMacro(OrthantProjection,bool);
  itkBooleanMacro(OrthantProjection);

  itkGetConstMacro(CurrentIteration, SizeValueType);
  itkGetConstReferenceMacro(Value, MeasureType);

  itkGetConstReferenceMacro(BestPosition, ParametersType);


protected:
  AdaptiveStepGradientDescentOptimizer();
  virtual ~AdaptiveStepGradientDescentOptimizer() {}
  virtual void PrintSelf(std::ostream & os, Indent indent) const ITK_OVERRIDE;


  double m_Fmax;
  double m_Fmin;
  double m_omega;
  double m_CurrentTime;
  double m_BestCurrentTime;
  double m_CurrentStep;
  double m_Previous_a;
  DerivativeType m_PreviousGradient;
  DerivativeType m_BestGradient;
  DerivativeType m_BestPreviousGradient;

  bool m_OrthantProjection;

  bool               m_Stop;
  MeasureType        m_Value;
  MeasureType        m_BestValue;
  ParametersType     m_BestPosition;
  StopConditionType  m_StopCondition;
  SizeValueType      m_CurrentIteration;
  std::ostringstream m_StopConditionDescription;

private:
  AdaptiveStepGradientDescentOptimizer(const Self &) = delete;
  void operator=(const Self &) = delete; 
};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkAdaptiveStepGradientDescentOptimizer.hxx"
#endif
