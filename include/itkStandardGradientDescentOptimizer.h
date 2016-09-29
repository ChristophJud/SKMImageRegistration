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
#include "itkGradientDescentOptimizer.h"
#include <string>
namespace itk
{
/** \class StandardGradientDescentOptimizer
 *
 * \ingroup Numerics Optimizers
 * \ingroup ITKOptimizers
 */
class StandardGradientDescentOptimizer:
  public GradientDescentOptimizer
{
public:
  /** Standard class typedefs. */
  typedef StandardGradientDescentOptimizer      Self;
  typedef GradientDescentOptimizer              Superclass;
  typedef SmartPointer< Self >                  Pointer;
  typedef SmartPointer< const Self >            ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(StandardGradientDescentOptimizer, GradientDescentOptimizer);



  /** Advance one step following the gradient direction. */
  virtual void AdvanceOneStep();
  itkGetConstMacro(a, double);
  void Seta(double a){
      if(a>0){
          m_a = a;
      }
      else{
          itkExceptionMacro("a has to be positive.")
      }
  }

  itkGetConstMacro(A, double);
  void SetA(double A){
      if(A>=1){
          m_A = A;
      }
      else{
          itkExceptionMacro("A has to be greather or equalt to 1.")
      }
  }

  itkGetConstMacro(alpha, double);
  void Setalpha(double alpha){
      if(0<alpha && alpha <=1){
          m_alpha = alpha;
      }
      else{
          itkExceptionMacro("alpha has to be in between 0 and 1 or 1.")
      }
  }

  virtual DerivativeType GetGradient(){
      return this->m_Gradient;
  }

protected:
  StandardGradientDescentOptimizer();
  virtual ~StandardGradientDescentOptimizer() {}
  virtual void PrintSelf(std::ostream & os, Indent indent) const ITK_OVERRIDE;

  double m_a;
  double m_A;
  double m_alpha;

private:
  StandardGradientDescentOptimizer(const Self &) = delete;
  void operator=(const Self &) = delete;


};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkStandardGradientDescentOptimizer.hxx"
#endif
