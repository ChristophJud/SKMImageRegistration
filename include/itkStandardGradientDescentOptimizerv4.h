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

#include "itkGradientDescentOptimizerv4.h"

namespace itk
{

template<typename TInternalComputationValueType>
class StandardGradientDescentOptimizerv4:
    public GradientDescentOptimizerv4Template<TInternalComputationValueType>
{
public:
    typedef StandardGradientDescentOptimizerv4    Self;
    typedef GradientDescentOptimizerv4Template<TInternalComputationValueType>
                                                Superclass;
    typedef SmartPointer< Self >                Pointer;
    typedef SmartPointer< const Self >          ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro(StandardGradientDescentOptimizerv4, Superclass);

    /** used superclass types */
    typedef typename Superclass::IndexRangeType     IndexRangeType;
    typedef typename Superclass::MetricType         MetricType;
    typedef typename MetricType::Pointer            MetricTypePointer;
    typedef typename Superclass::DerivativeType     DerivativeType;
    typedef typename Superclass::MeasureType        MeasureType;
    typedef typename Superclass::ScalesType         ScalesType;
    typedef typename Superclass::ParametersType     ParametersType;


    /** Get/Set methods for Standard Gradient Descent specific variables */
    itkGetConstMacro(a, double);
    void Seta(double a){
        if(a>0)
            m_a = a;
        else
            itkExceptionMacro("a has to be positive.")
    }

    itkGetConstMacro(A, double);
    void SetA(double A){
        if(A>=1)
            m_A = A;
        else
            itkExceptionMacro("A has to be greather or equalt to 1.")
    }

    itkGetConstMacro(alpha, double);
    void Setalpha(double alpha){
        if(0<alpha && alpha <=1)
            m_alpha = alpha;
        else
            itkExceptionMacro("alpha has to be in between 0 and 1 or 1.")
    }

    itkGetConstMacro(OrthantProjection,bool);
    itkSetMacro(OrthantProjection,bool);
    itkBooleanMacro(OrthantProjection);

    itkGetConstMacro(CalculateMagnitude,bool);
    itkSetMacro(CalculateMagnitude,bool);
    itkBooleanMacro(CalculateMagnitude);

    itkGetConstMacro(CountZeroParams,bool);
    itkSetMacro(CountZeroParams,bool);
    itkBooleanMacro(CountZeroParams);

    /** Estimate the learning rate based on the current gradient. */
    virtual void EstimateLearningRate() ITK_OVERRIDE;

protected:
    StandardGradientDescentOptimizerv4();
    virtual ~StandardGradientDescentOptimizerv4() {}

    virtual void PrintSelf(std::ostream & os, Indent indent) const ITK_OVERRIDE;

    virtual void ModifyGradientByLearningRateOverSubRange( const IndexRangeType& subrange ) ITK_OVERRIDE;

    double m_a;
    double m_A;
    double m_alpha;

    bool m_OrthantProjection;

    bool   m_CalculateMagnitude;
    bool   m_CountZeroParams;

private:
    StandardGradientDescentOptimizerv4(const Self &) = delete;
    void operator=(const Self &) = delete;

};

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkStandardGradientDescentOptimizerv4.hxx"
#endif