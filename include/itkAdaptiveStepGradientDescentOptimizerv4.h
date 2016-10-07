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
class AdaptiveStepGradientDescentOptimizerv4:
    public StandardGradientDescentOptimizerv4<TInternalComputationValueType>
{
public:
    typedef AdaptiveStepGradientDescentOptimizerv4  Self;
    typedef StandardGradientDescentOptimizerv4<TInternalComputationValueType>
                                                    Superclass;
    typedef SmartPointer< Self >                    Pointer;
    typedef SmartPointer< const Self >              ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro(AdaptiveStepGradientDescentOptimizerv4, Superclass);

    /** Superclass typedefs */
    typedef typename Superclass::IndexRangeType     IndexRangeType;
    typedef typename Superclass::MetricType         MetricType;
    typedef typename MetricType::Pointer            MetricTypePointer;
    typedef typename Superclass::DerivativeType     DerivativeType;
    typedef typename Superclass::MeasureType        MeasureType;
    typedef typename Superclass::ScalesType         ScalesType;
    typedef typename Superclass::ParametersType     ParametersType;

    typedef TInternalComputationValueType           InternalComputationValueType;

    itkGetConstMacro(Fmax, double);
    void SetFmax(double fmax){
        if(fmax>0)
            m_Fmax = fmax;
        else
            itkExceptionMacro("F_max has to be positive");
    }

    itkGetConstMacro(Fmin, double);
    void SetFmin(double fmin){
        if(fmin<0)
            m_Fmin = fmin;
        else
            itkExceptionMacro("F_min ha to be negative")
    }

    itkGetConstMacro(omega, double);
    void Setomega(double omega){
        if(omega>0)
            m_omega = omega;
        else
            itkExceptionMacro("omega has to be positive.")
    }

    virtual void StartOptimization( bool doOnlyInitialization = false ) ITK_OVERRIDE;

    /** Estimate the learning rate based on the current gradient. */
    virtual void EstimateLearningRate() ITK_OVERRIDE;
    virtual InternalComputationValueType CalculateScalarProduct(const IndexRangeType& subrange ) const;
    virtual void ModifyCurrentTime(InternalComputationValueType scalarproduct );

protected:
    AdaptiveStepGradientDescentOptimizerv4();
    virtual ~AdaptiveStepGradientDescentOptimizerv4() {}

    virtual void PrintSelf(std::ostream & os, Indent indent) const ITK_OVERRIDE;

    virtual void ModifyGradientByLearningRateOverSubRange( const IndexRangeType& subrange ) ITK_OVERRIDE;

    double m_Fmax;
    double m_Fmin;
    double m_omega;
    double m_CurrentTime;

    DerivativeType m_TemporaryGradient;

private:
    AdaptiveStepGradientDescentOptimizerv4(const Self &) = delete;
    void operator=(const Self &) = delete;

};

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkAdaptiveStepGradientDescentOptimizerv4.hxx"
#endif