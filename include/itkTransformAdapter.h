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

#include <itkTransform.h>
#include <itkImage.h>
#include <itkVector.h>
#include <itkSymmetricSecondRankTensor.h>

#include "CommonTypes.h"
#include "GpuEvaluator.h"

namespace itk{

template< typename TScalar, unsigned int NDimensions >
class TransformAdapter : public Transform< TScalar, NDimensions, NDimensions >{
public:
	/** standard class typedefs. */
	typedef TransformAdapter                                Self;
	typedef Transform< TScalar, NDimensions, NDimensions > 	Superclass;
	typedef SmartPointer< Self >                           	Pointer;
	typedef SmartPointer< const Self >                     	ConstPointer;

	/** re-typdefing */
	typedef typename Superclass::ParametersType 			ParametersType;
	typedef typename Superclass::FixedParametersType 		FixedParametersType;
	typedef typename Superclass::JacobianType 				JacobianType;
	typedef typename Superclass::InputPointType  			InputPointType;
	typedef typename Superclass::OutputPointType 			OutputPointType;

    /** standard itk macros */
	itkTypeMacro(TransformAdapter, Transform);                      // Run-time type information (and related methods).
	itkNewMacro(Self);                                              // New macro for creation of through a Smart Pointer
	itkStaticConstMacro(SpaceDimension, unsigned int, NDimensions); // Dimension of the domain space.

    /** image and control point image typedefs */
	typedef Image<TScalar, NDimensions>  							ImageType;
	typedef Image<Vector<TScalar, NDimensions>, NDimensions >   	ControlPointImageType;
	typedef Image<TScalar, NDimensions >   							ControlPointWeightImageType;
	typedef Image<TScalar, NDimensions>								WeightImageType;
	typedef Image<SymmetricTensorType::EigenVectorsMatrixType, 
					SpaceDimensions>    							TensorImageType;
	
	typedef TensorImageType::PixelType 								CovarianceMatrixType;

	typedef std::unique_ptr<GpuKernelEvaluator>						GpuKernelEvaluatorPointer;

	/** get/set macros */
	itkGetConstMacro(SigmaC0, 	double);
	itkSetMacro		(SigmaC0, 	double);
	itkGetConstMacro(SigmaC4, 	double);
	itkSetMacro		(SigmaC4, 	double);
	itkGetConstMacro(Subsample, unsigned);
	itkSetMacro		(Subsample, unsigned);
	itkGetConstMacro(SubsampleNeighborhood, unsigned);
	itkSetMacro		(SubsampleNeighborhood, unsigned);
	itkGetConstMacro(UseWeightImage,	    bool);
    itkGetConstMacro(UseWeightTensor,	    bool);
    itkGetConstMacro(MaximumWeight,		    double);
    itkGetConstMacro(MaximumEigenvalueCovariance,CovarianceMatrixType);

	/** set the transform parameters and update internal transformation */
	virtual void SetParameters(const ParametersType &) ITK_OVERRIDE;

	/** update the parameters array */
	virtual void UpdateParameters() const;

	/** get the transformation parameters */
	virtual const ParametersType & GetParameters() const ITK_OVERRIDE;

	/** Sets the value pv of the control point which is nearest to the physical point p */
	void SetParameter(typename ControlPointImageType::PointType p, typename ControlPointImageType::PixelType pv);
	
	/** Sets the value pv of the control point with index idx */
	void SetParameter(typename ControlPointImageType::IndexType idx, typename ControlPointImageType::PixelType pv);

	/** get parameters as contol point image */
	typename ControlPointImageType::Pointer GetControlPointImage()const;

	/** set parameters by providing a control point image */
	void SetControlPointImage(const typename ControlPointImageType::Pointer image);

	/** get parameter weights as control point weighting image */
	const typename ControlPointWeightImageType::Pointer GetControlPointWeightImage()const;

	/** set weights on parameters by providing a weight image */
	void SetControlPointWeightImage(const typename ControlPointWeightImageType::Pointer weight_image);

	void ComputeControlPointImage(const typename ImageType::Pointer reference_image, typename ControlPointImageType::SpacingType sigma);

    /** Index mapping from control point image indexing to flat indexing of parameter array */
    unsigned long GetFlatIndex(const typename ControlPointImageType::IndexType& current_index, unsigned dimension)const;

	/** get the fixed transformation parameters */
	virtual const FixedParametersType & GetFixedParameters() const ITK_OVERRIDE;

	/** set the fixed transformation parameters, which are not present in this adapter */
	virtual void SetFixedParameters(const FixedParametersType &) ITK_OVERRIDE{}

	/** functions which have to be provided, but are currently not needed */
	virtual OutputPointType TransformPoint(const InputPointType & thisPoint) const ITK_OVERRIDE;
	
  	virtual void ComputeJacobianWithRespectToParameters( const InputPointType  & point, JacobianType & jacobian) const ITK_OVERRIDE{
		  itkExceptionMacro( "ComputeJacobianWithRespectToParameters not yet implemented for " << this->GetNameOfClass() );
	}

	/** set reference and target images */
	void SetReferenceImage(ImageType * referenceImage);
	void SetTargetImage   (ImageType * targetImage);

	/** sets the weight image */
	void SetWeightingImage(const typename ImageType::Pointer weight_image);

	/** sets the tensor image */
	void SetWeightingTensor(const typename TensorImageType::Pointer weight_tensor, TScalar maximum_tilt=-1);

	/** initialization mainly of the GPU evaluator. has to be called befor first usage */
	void Initialize();

	GpuKernelEvaluator* GetGpuKernelEvaluator() const;

protected:
    TransformAdapter();
    virtual ~TransformAdapter(){};

	double m_SigmaC0;
	double m_SigmaC4;

	typename ControlPointImageType::Pointer 		m_CPImage;
	typename ControlPointImageType::SpacingType 	m_CPSpacing;
	typename ControlPointWeightImageType::Pointer 	m_CPWImage;

	typename WeightImageType::Pointer 				m_WeightImage;
	typename TensorImageType::Pointer 				m_WeightTensor;
	bool 	m_UseWeightImage;
	bool 	m_UseWeightTensor;

	double  m_MaximumWeight;
	CovarianceMatrixType m_MaximumEigenvalueCovariance;

	typename ImageType::Pointer						m_ReferenceImage;
	typename ImageType::Pointer						m_TargetImage;
	unsigned m_Subsample;
	unsigned m_SubsampleNeighborhood;

	GpuKernelEvaluatorPointer kernelEvaluator;
	
private:
	TransformAdapter(const Self &) = delete;
  	void operator=(const Self &) = delete;
};

}

#include "itkTransformAdapter.hxx"