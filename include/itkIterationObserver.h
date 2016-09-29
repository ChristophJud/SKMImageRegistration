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
#include <chrono>

#include "itkCommand.h"
#include "itkWeakPointer.h"

namespace itk {

/**
 *  Implementation of the Command Pattern to be invoked every iteration
 * \class IterationObserver
 * \ingroup ITKRegistrationCommon
 */
template < typename TOptimizer >
class IterationObserver : public Command
{
public:

    typedef IterationObserver   Self;
    typedef itk::Command  Superclass;
    typedef itk::SmartPointer<Self>  Pointer;
    typedef itk::SmartPointer<const Self>  ConstPointer;

    /**
   * Execute method will print data at each iteration
   */
    virtual void Execute(itk::Object *caller, const itk::EventObject & event) ITK_OVERRIDE{
        Execute( (const itk::Object *)caller, event);
    }

    virtual void Execute(const itk::Object *, const itk::EventObject & event) ITK_OVERRIDE{
        if( typeid( event ) == typeid( itk::StartEvent ) ){
            std::cout << std::endl << "Position              Value";
            std::cout << std::endl << std::endl;
        }
        else if( typeid( event ) == typeid( itk::IterationEvent ) ){                       
            std::cout << " \tIteration: " << m_Optimizer->GetCurrentIteration() << std::flush;
            if(m_PrintValue)std::cout << ", metric: " << m_Optimizer->GetValue() << std::flush;
            if(m_PrintPosition)std::cout << ", position: " << m_Optimizer->GetCurrentPosition() << std::flush;

            /// time measuring
            std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = now-last_time;
            std::cout << " \t(" << elapsed_seconds.count() << "s)" << std::flush;
            last_time = now;
            std::cout << std::endl;
        }
        else if( typeid( event ) == typeid( itk::EndEvent ) ){
            std::cout << std::endl << std::endl;
            std::cout << "After " << m_Optimizer->GetCurrentIteration();
            std::cout << "  iterations " << std::endl;
            std::cout << "Solution is    = " << m_Optimizer->GetCurrentPosition();
            std::cout << std::endl;
            std::cout << "With value     = " << m_Optimizer->GetValue();
            std::cout << std::endl;
            std::cout << "Stop condition = " << m_Optimizer->GetStopCondition();
            std::cout << std::endl;
        }

    }

    itkTypeMacro( IterationObserver, ::itk::Command );
    itkNewMacro( Self );

    itkSetMacro(PrintPosition, bool);
    itkGetConstMacro(PrintPosition, bool);
    itkBooleanMacro(PrintPosition);

    itkSetMacro(PrintValue, bool);
    itkGetConstMacro(PrintValue, bool);
    itkBooleanMacro(PrintValue);

    typedef    TOptimizer     OptimizerType;

    void SetOptimizer( OptimizerType * optimizer ){
        m_Optimizer = optimizer;
        m_Optimizer->AddObserver( itk::IterationEvent(), this );
    }

protected:
    IterationObserver():m_PrintPosition(true), m_PrintValue(true), last_time(std::chrono::system_clock::now()) {};

private:
    WeakPointer<OptimizerType>   m_Optimizer;
    bool m_PrintPosition;
    bool m_PrintValue;

    std::chrono::time_point<std::chrono::system_clock> last_time;
};

} // end namespace itk
