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

#include "itkCompositeTransform.h"

#include <deque>

namespace itk{

/** \class SummationTransform
 * \brief This class contains a list of transforms and concatenates them by addition.

 *
 * \ingroup ITKTransform
 */
template
<class TScalar = double, unsigned int NDimensions = 3>
class SummationTransform :
  public CompositeTransform<TScalar, NDimensions>
{
public:
  /** Standard class typedefs. */
  typedef SummationTransform                                  Self;
  typedef CompositeTransform<TScalar, NDimensions>   Superclass;
  typedef SmartPointer<Self>                                  Pointer;
  typedef SmartPointer<const Self>                            ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro( SummationTransform, Transform );

  /** New macro for creation of through a Smart Pointer */
  itkNewMacro( Self );

  /** Sub transform type **/
  typedef typename Superclass::TransformType                TransformType;
  typedef typename Superclass::TransformTypePointer         TransformTypePointer;
  /** InverseTransform type. */
  typedef typename Superclass::InverseTransformBasePointer  InverseTransformBasePointer;
  /** Scalar type. */
  typedef typename Superclass::ScalarType                 ScalarType;
  /** Parameters type. */
  typedef typename Superclass::ParametersType             ParametersType;
  typedef typename Superclass::ParametersValueType        ParametersValueType;
  /** Derivative type */
  typedef typename Superclass::DerivativeType             DerivativeType;
  /** Jacobian type. */
  typedef typename Superclass::JacobianType               JacobianType;
  /** Transform category type. */
  typedef typename Superclass::TransformCategoryType      TransformCategoryType;
  /** Standard coordinate point type for this class. */
  typedef typename Superclass::InputPointType             InputPointType;
  typedef typename Superclass::OutputPointType            OutputPointType;
  /** Standard vector type for this class. */
  typedef typename Superclass::InputVectorType            InputVectorType;
  typedef typename Superclass::OutputVectorType           OutputVectorType;
  /** Standard covariant vector type for this class */
  typedef typename Superclass::InputCovariantVectorType   InputCovariantVectorType;
  typedef typename Superclass::OutputCovariantVectorType  OutputCovariantVectorType;
  /** Standard vnl_vector type for this class. */
  typedef typename Superclass::InputVnlVectorType         InputVnlVectorType;
  typedef typename Superclass::OutputVnlVectorType        OutputVnlVectorType;
  /** Standard Vectorpixel type for this class */
  typedef typename Superclass::InputVectorPixelType       InputVectorPixelType;
  typedef typename Superclass::OutputVectorPixelType      OutputVectorPixelType;
  /** Standard DiffusionTensor3D typedef for this class */
  typedef typename Superclass::InputDiffusionTensor3DType  InputDiffusionTensor3DType;
  typedef typename Superclass::OutputDiffusionTensor3DType OutputDiffusionTensor3DType;
  /** Standard SymmetricSecondRankTensor typedef for this class */
  typedef typename Superclass::InputSymmetricSecondRankTensorType   InputSymmetricSecondRankTensorType;
  typedef typename Superclass::OutputSymmetricSecondRankTensorType  OutputSymmetricSecondRankTensorType;

  /** Transform queue type */
  typedef typename Superclass::TransformQueueType         TransformQueueType;

  /** The number of parameters defininig this transform. */
  typedef typename Superclass::NumberOfParametersType     NumberOfParametersType;

  /** Optimization flags queue type */
  typedef std::deque<bool>                                TransformsToOptimizeFlagsType;

  /** Dimension of the domain spaces. */
  itkStaticConstMacro( InputDimension, unsigned int, NDimensions );
  itkStaticConstMacro( OutputDimension, unsigned int, NDimensions );

  /** Compute the position of point in the new space.
  *
  * Transforms are applied starting from the *back* of the
  * queue. That is, in reverse order of which they were added, in order
  * to work properly with ResampleFilter.
  *
  * Imagine a user wants to apply an Affine transform followed by a Deformation
  * Field (DF) transform. He adds the Affine, then the DF. Because the user
  * typically conceptualizes a transformation as being applied from the Moving
  * image to the Fixed image, this makes intuitive sense. But since the
  * ResampleFilter expects to transform from the Fixed image to the Moving
  * image, the transforms are applied in reverse order of addition, i.e. from
  * the back of the queue, and thus, DF then Affine.
  */
  virtual OutputPointType TransformPoint( const InputPointType & inputPoint ) const ITK_OVERRIDE;

  /**
   * Compute the Jacobian with respect to the parameters for the compositie
   * transform using Jacobian rule. See comments in the implementation.
   */
  virtual void ComputeJacobianWithRespectToParameters(const InputPointType  & p, JacobianType & j) const ITK_OVERRIDE;

protected:
  SummationTransform();
  virtual ~SummationTransform();
  virtual void PrintSelf( std::ostream& os, Indent indent ) const ITK_OVERRIDE;

private:
  SummationTransform( const Self & ) = delete;
  void operator=( const Self & ) = delete;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkSummationTransform.hxx"
#endif
