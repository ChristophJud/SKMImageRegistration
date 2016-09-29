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

#include "itkSummationTransform.h"

namespace itk
{


template
<typename TScalar, unsigned int NDimensions>
SummationTransform<TScalar, NDimensions>::SummationTransform() : Superclass()
{
}


template
<typename TScalar, unsigned int NDimensions>
SummationTransform<TScalar, NDimensions>::
~SummationTransform()
{
}



template
<typename TScalar, unsigned int NDimensions>
typename SummationTransform<TScalar, NDimensions>
::OutputPointType
SummationTransform<TScalar, NDimensions>
::TransformPoint( const InputPointType& inputPoint ) const
{
    OutputPointType outputPoint( inputPoint );

    typename TransformQueueType::const_iterator it;
    /* Apply in reverse queue order.  */
    it = this->m_TransformQueue.end();

    do
    {
        it--;
        outputPoint = (*it)->TransformPoint( outputPoint );
    }
    while( it != this->m_TransformQueue.begin() );

    return outputPoint;
}

template <typename TScalar, unsigned int NDimensions>
void
SummationTransform<TScalar, NDimensions>
::ComputeJacobianWithRespectToParameters( const InputPointType & p, JacobianType & outJacobian ) const
{
    /* Returns a concatenated MxN array, holding the Jacobian of each sub
   * transform that is selected for optimization. The order is the same
   * as that in which they're applied, i.e. reverse order.
   * M rows = dimensionality of the transforms
   * N cols = total number of parameters in the selected sub transforms. */
    outJacobian.SetSize( NDimensions, this->GetNumberOfLocalParameters() );

    NumberOfParametersType offset = NumericTraits< NumberOfParametersType >::ZeroValue();

    signed long tind = (signed long) this->GetNumberOfTransforms() - 1;
    /* Get a raw pointer for efficiency, avoiding SmartPointer register/unregister */
    const TransformType * const transform = this->GetNthTransformConstPointer( tind );


    if( this->GetNthTransformToOptimize( tind ) ){
        /* Copy from another matrix, element-by-element */
        /* The matrices are row-major, so block copy is less obviously
       * better */

        const NumberOfParametersType numberOfLocalParameters = transform->GetNumberOfLocalParameters();

        typename TransformType::JacobianType current_jacobian( NDimensions, numberOfLocalParameters );
        transform->ComputeJacobianWithRespectToParameters( p, current_jacobian );
        outJacobian.update( current_jacobian, 0, offset );
        offset += numberOfLocalParameters;
    }
    else{
        itkExceptionMacro("Unexpected error in calculating jacobian of summation transform.");
    }

}



template <typename TScalar, unsigned int NDimensions>
void
SummationTransform<TScalar, NDimensions>
::PrintSelf( std::ostream& os, Indent indent ) const
{
    Superclass::PrintSelf( os, indent );
}


} // end namespace itk

