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

#include <cmath>

#include "itkUnaryFunctorImageFilter.h"
#include "itkSymmetricEigenAnalysis.h"
#include "itkSymmetricSecondRankTensor.h"
#include "itkVector.h"
#include "itkMatrix.h"

namespace itk
{
// This functor class invokes the computation of Eigen Analysis for
// every pixel. The input pixel type must provide the API for the [][]
// operator, while the output pixel type must provide the API for the
// [] operator. Input pixel matrices should be symmetric.
//
// The default operation is to order eigen values in ascending order.
// You may also use OrderEigenValuesBy( ) to order eigen values by
// magnitude as is common with use of tensors in vessel extraction.
namespace Functor
{
template< typename TInputMatrix, typename TVectorType, typename TEigenMatrixType >
class SymmetricEigenDecompositionFunction
{
public:
  typedef typename TInputMatrix::ValueType RealValueType;
  SymmetricEigenDecompositionFunction() {}
  ~SymmetricEigenDecompositionFunction() {}
  typedef SymmetricEigenAnalysis< TInputMatrix, TVectorType, TEigenMatrixType > CalculatorType;
  bool operator!=(const SymmetricEigenDecompositionFunction &) const
  {
    return false;
  }

  bool operator==(const SymmetricEigenDecompositionFunction & other) const
  {
    return !( *this != other );
  }

  inline TEigenMatrixType operator()(const TInputMatrix & x) const
  {
    typename CalculatorType::VectorType eigenValues;
    typename CalculatorType::EigenMatrixType eigenVectors;

    m_Calculator.ComputeEigenValuesAndVectors(x, eigenValues, eigenVectors);

    // multiply eigenvectors with eigenvalues and return eigenVectors
    // eigenVectors are row-wise

    for(unsigned i=0; i<m_Calculator.GetDimension(); i++){
        for(unsigned j=0; j<m_Calculator.GetDimension(); j++){
            eigenVectors[i][j] *= eigenValues[i];
        }
    }
    return eigenVectors;
  }

  /** Method to explicitly set the dimension of the matrix */
  void SetDimension(unsigned int n)
  {
    m_Calculator.SetDimension(n);
  }
  unsigned int GetDimension() const
  {
    return m_Calculator.GetDimension();
  }

  /** Typdedefs to order eigen values.
   * OrderByValue:      lambda_1 < lambda_2 < ....
   * OrderByMagnitude:  |lambda_1| < |lambda_2| < .....
   * DoNotOrder:        Default order of eigen values obtained after QL method
   */
  typedef enum {
    OrderByValue = 1,
    OrderByMagnitude,
    DoNotOrder
    } EigenValueOrderType;

  /** Order eigen values. Default is to OrderByValue:  lambda_1 <
   * lambda_2 < .... */
  void OrderEigenValuesBy(EigenValueOrderType order)
  {
    if ( order == OrderByValue )
      {
      m_Calculator.SetOrderEigenValues(true);
      }
    else if ( order == OrderByMagnitude )
      {
      m_Calculator.SetOrderEigenMagnitudes(true);
      }
    else if ( order == DoNotOrder )
      {
      m_Calculator.SetOrderEigenValues(false);
      }
  }

private:
  CalculatorType m_Calculator;
};
}  // end namespace functor

/** \class SymmetricEigenDecompositionImageFilter
 * \brief Computes the eigen-values of every input symmetric matrix pixel.
 *
 * SymmetricEigenDecompositionImageFilter applies pixel-wise the invokation for
 * computing the eigen-values and eigen-vectors of the symmetric matrix
 * corresponding to every input pixel.
 *
 * The OrderEigenValuesBy( .. ) method can be used to order eigen values
 * in ascending order by value or magnitude or no ordering.
 * OrderByValue:      lambda_1 < lambda_2 < ....
 * OrderByMagnitude:  |lambda_1| < |lambda_2| < .....
 * DoNotOrder:        Default order of eigen values obtained after QL method
 *
 * The user of this class is explicitly supposed to set the dimension of the
 * 2D matrix using the SetDimension() method.
 *
 * \ingroup IntensityImageFilters  MultiThreaded  TensorObjects
 *
 * \ingroup ITKImageIntensity
 */
template< typename  TInputImage, typename  TOutputImage >
class SymmetricEigenDecompositionImageFilter:
  public
  UnaryFunctorImageFilter< TInputImage, TOutputImage,
                           Functor::SymmetricEigenDecompositionFunction<
                             typename TInputImage::PixelType, // MatrixType
                             itk::FixedArray< typename TInputImage::PixelType::ComponentType, TInputImage::ImageDimension >, // EigenValuesArrayType
                             typename TInputImage::PixelType > > // EigenVectorsMatrixType
{
public:
  /** Standard class typedefs. */
  typedef SymmetricEigenDecompositionImageFilter Self;
  typedef UnaryFunctorImageFilter<
    TInputImage, TOutputImage,
    Functor::SymmetricEigenDecompositionFunction<
        typename TInputImage::PixelType,
        itk::FixedArray<typename TInputImage::PixelType::ComponentType, TInputImage::ImageDimension >,
        typename TInputImage::PixelType > >   Superclass;

  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  typedef typename Superclass::OutputImageType OutputImageType;
  typedef typename TOutputImage::PixelType     OutputPixelType;
  typedef typename TInputImage::PixelType      InputPixelType;
  typedef typename InputPixelType::ValueType   InputValueType;
  typedef typename Superclass::FunctorType     FunctorType;

  /** Typdedefs to order eigen values.
   * OrderByValue:      lambda_1 < lambda_2 < ....
   * OrderByMagnitude:  |lambda_1| < |lambda_2| < .....
   * DoNotOrder:        Default order of eigen values obtained after QL method
   */
  typedef typename FunctorType::EigenValueOrderType EigenValueOrderType;

  /** Order eigen values. Default is to OrderByValue:  lambda_1 <
   * lambda_2 < .... */
  void OrderEigenValuesBy(EigenValueOrderType order)
  {
    this->GetFunctor().OrderEigenValuesBy(order);
  }

  /** Run-time type information (and related methods).   */
  itkTypeMacro(SymmetricEigenDecompositionImageFilter, UnaryFunctorImageFilter);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Print internal ivars */
  void PrintSelf(std::ostream & os, Indent indent) const ITK_OVERRIDE
  { this->Superclass::PrintSelf(os, indent); }

  /** Set the dimension of the tensor. (For example the SymmetricSecondRankTensor
   * is a pxp matrix) */
  void SetDimension(unsigned int p)
  {
    this->GetFunctor().SetDimension(p);
  }
  unsigned int GetDimension() const
  {
    return this->GetFunctor().GetDimension();
  }

#ifdef ITK_USE_CONCEPT_CHECKING
  // Begin concept checking
  itkConceptMacro( InputHasNumericTraitsCheck,
                   ( Concept::HasNumericTraits< InputValueType > ) );
  // End concept checking
#endif

protected:
  SymmetricEigenDecompositionImageFilter() {}
  virtual ~SymmetricEigenDecompositionImageFilter() {}

private:
  SymmetricEigenDecompositionImageFilter(const Self &) ITK_DELETE_FUNCTION;
  void operator=(const Self &) ITK_DELETE_FUNCTION;
};
} // end namespace itk
