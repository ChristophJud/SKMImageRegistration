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

#include <itkImageToImageMetric.h>
#include <itkDisplacementFieldTransform.h>

#include "CommonTypes.h"
#include "itkSummationTransform.h"
#include "itkTransformAdapter.h"

namespace itk{

template< typename TFixedImage, typename TMovingImage >
class RegularizedImageToImageMetric: public ImageToImageMetric< TFixedImage, TMovingImage >{
public:
    /** standard class typedefs. */
    typedef RegularizedImageToImageMetric                   Self;
    typedef ImageToImageMetric< TFixedImage, TMovingImage > Superclass;
    typedef SmartPointer< Self >                            Pointer;
    typedef SmartPointer< const Self >                      ConstPointer;

    /** typedefs of superclass */
    typedef typename Superclass::TransformType                TransformType;
    typedef typename Superclass::MeasureType                  MeasureType;
    typedef typename Superclass::DerivativeType               DerivativeType;
    typedef typename Superclass::ParametersType               ParametersType;
    typedef typename Superclass::ParametersValueType          ParametersValueType;

    /** typedefs for vector of points */
    typedef typename Superclass::MovingImagePointType         MovingImagePointType;
    typedef typename std::vector<MovingImagePointType>        PointListType;

    /** typedefs for supported transforms */
    typedef SummationTransform<ScalarType, SpaceDimensions>             SummationTransformType;
    typedef TransformAdapter<ScalarType, SpaceDimensions>               TransformAdapterType;  

    /** typedef for preceeding transform in case of using the summation transform */
    typedef DisplacementFieldTransform<ScalarType, SpaceDimensions>     DisplacementFieldTransformType;    

    /** for callbacks */
    typedef std::vector<double> TemporaryValuesType;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro(RegularizedImageToImageMetric, ImageToImageMetric); 

    /** regularization get/set macros */
    itkGetConstMacro(RegularizerL1,double);
    itkSetMacro     (RegularizerL1,double);
    itkGetConstMacro(RegularizerL21,double);
    itkSetMacro     (RegularizerL21,double);
    itkGetConstMacro(RegularizerL2,double);
    itkSetMacro     (RegularizerL2,double);
    itkGetConstMacro(RegularizerRKHS,double);
    itkSetMacro     (RegularizerRKHS,double);
    itkGetConstMacro(RegularizerRD,double);
    itkSetMacro     (RegularizerRD,double);
    itkGetConstMacro(RegularizerPG,double);
    itkSetMacro     (RegularizerPG,double);
    itkGetConstMacro(RegularizerRDScaling,double);
    itkSetMacro     (RegularizerRDScaling,double);
    itkGetConstMacro(RegularizerPGScaling,double);
    itkSetMacro     (RegularizerPGScaling,double);

    /** switch on/off NCC loss function (by default MSE is used) */
    itkGetConstMacro(UseNccMetric,bool);
    itkSetMacro     (UseNccMetric,bool);
    itkBooleanMacro (UseNccMetric);

    /** switch on/off LCC loss function (by default MSE is used) */
    itkGetConstMacro(UseLccMetric,bool);
    itkSetMacro     (UseLccMetric,bool);
    itkBooleanMacro (UseLccMetric);

    /** switch on/off resampling before evaluation **/
    itkGetConstMacro(DoResampling,bool);
    itkSetMacro     (DoResampling,bool);
    itkBooleanMacro (DoResampling);

    /** rate of subsampling get/set */
    itkGetConstMacro(SubSample,unsigned);
    itkSetMacro     (SubSample,unsigned);
    itkGetConstMacro(SubSampleNeighborhood,unsigned);
    itkSetMacro     (SubSampleNeighborhood,unsigned);

    /** further get/set methods */
    itkGetConstMacro(OutputFilename,std::string);
    itkSetMacro     (OutputFilename,std::string);
    itkGetConstMacro(Verbosity,unsigned);
    itkSetMacro     (Verbosity,unsigned);

    /** methods which have to be implemented */
    virtual void Initialize(void) throw ( ExceptionObject ) ITK_OVERRIDE;

    /** get metric value (calls GetValueAndDerivative). */
	MeasureType GetValue(const ParametersType & parameters) const ITK_OVERRIDE;

	/** get metric derivative (calls GetValueAndDerivative). */
	void GetDerivative(const ParametersType & parameters,
                     DerivativeType & Derivative) const ITK_OVERRIDE;

	/**  Get the value and derivatives for single valued optimizers. */
	void GetValueAndDerivative(const ParametersType & parameters,
                             MeasureType & Value,
                             DerivativeType & Derivative) const ITK_OVERRIDE;

    const PointListType& GetFixedLandmarks()const{ return m_FixedLandmarks; }
    const PointListType& GetMovingLandmarks()const{ return m_MovingLandmarks; }

    void SetFixedLandmarks(const PointListType& landmarks) { m_FixedLandmarks = landmarks; }
    void SetMovingLandmarks(const PointListType& landmarks) { m_MovingLandmarks = landmarks; }

    MeasureType GetTRE() const;

    template<typename Printer>
    void SetPrintFunction(Printer&& p);

protected:
    RegularizedImageToImageMetric();
    virtual ~RegularizedImageToImageMetric();

    unsigned    m_SubSample;
    unsigned    m_SubSampleNeighborhood;
    bool        m_UseNccMetric;
    bool        m_UseLccMetric;

    /** regularization weights (default weight: switched off by zero value) */
    double m_RegularizerL1;    // l1 regularization 
    double m_RegularizerL21;   // l2,1 regularization
    double m_RegularizerL2;    // l2 regularization
    double m_RegularizerRKHS;  // rkhs norm regularization (calculated on GPU)
    double m_RegularizerRD;    // radiometric differences regularization (calculated on GPU)
    double m_RegularizerPG;    // parallelogram regularization (calculated on GPU)

    /** the robust regularizers do have a scaling */
    double m_RegularizerRDScaling;
    double m_RegularizerPGScaling;

private:
  RegularizedImageToImageMetric(const Self &) = delete;
  RegularizedImageToImageMetric& operator=(const Self &) = delete;

  PointListType m_FixedLandmarks;
  PointListType m_MovingLandmarks;

  std::string   m_OutputFilename;
  unsigned      m_Verbosity;

  bool          m_DoResampling;

  // print out callback
  std::function<void(const TemporaryValuesType&)> printFunction; 
};

}
#include "itkRegularizedImageToImageMetric.hxx"