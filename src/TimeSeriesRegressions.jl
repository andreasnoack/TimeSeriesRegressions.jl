using Distributions, Stats, Civecm

module TimeSeriesRegressions

import Base:show, LinAlg.dot
import Distributions: loglikelihood
import Stats: coeftable, vcov
import Civecm: residuals

export adf, adl, ecm, ecmI2

# General methods
dot(x::AbstractVector) = dot(x,x)

# Abstract types
abstract UnivariateRegressionModel

## Methods
function vcov(obj::UnivariateRegressionModel)
	rinv = pinv(obj.predfact[:R])
	ipivots = invperm(obj.predfact[:p])
	return (rinv*rinv'*dot(residuals(obj))/length(obj.response))[ipivots, ipivots]
end

coeftable(obj::UnivariateRegressionModel) = [obj.parest obj.parest./sqrt(diag(vcov(obj)))]

function loglikelihood(obj::UnivariateRegressionModel)
	n = length(obj.response)
	return -0.5*n*(log(2π) + 1. + log(dot(residuals(obj))/n))
end

residuals(obj::UnivariateRegressionModel) = obj.response - obj.predictor*obj.parest

# ADL type
type ADL <: UnivariateRegressionModel
	endogenous::Vector{Float64}
	exogenous::Matrix{Float64}
	lags::Int64
	parest::Vector{Float64}
	response::Vector{Float64}
	predictor::Matrix{Float64}
	predfact::QRPivoted{Float64}
end

## Constructors

function adl(endogenous::Vector{Float64}, exogenous::Matrix{Float64}, lags::Integer)
	if lags < 0 error("Number of lags must be non-negative") end
	T = length(endogenous) - lags
	dimexo = size(exogenous, 2)
	if size(exogenous, 1) != T + lags error("Endogenous and exogenous variables must have same sample length") end
	response = Array(Float64, T)
	predictor = Array(Float64, T, lags + dimexo*(lags + 1))
	# z2 = Array(Float64, T, lags - 1)
	for i = 1:T
		response[i] = endogenous[i+lags]
		cl = 1
		for j = 1:lags
			predictor[i,cl] = endogenous[i+lags-j]
			cl += 1
		end
		if dimexo > 0
			for j = 0:lags
				for k = 1:dimexo
					predictor[i,cl] = exogenous[i+lags-j,k]
					cl += 1
				end
			end
		end
	end
	fit = regress(response, predictor)
	return ADL(endogenous, exogenous, lags, fit.parest, response, predictor, fit.predfact)
end
adl(endogenous::Vector{Float64}, lags::Integer) = adl(endogenous, zeros(length(endogenous),0), lags)

## Other methods

# LinearRegression type
type LinearRegression <: UnivariateRegressionModel
	response::Vector{Float64}
	predictor::Matrix{Float64}
	parest::Vector{Float64}
	predfact::QRPivoted{Float64}
end

## Constructors
function regress(y::StridedVector{Float64}, X::StridedMatrix{Float64})
	n = length(y)
	if size(X, 1) != n error("Dimensions don't match") end
	Xfact = qrpfact(X)
	if size(X, 2) == 0
		parest = zeros(0)
		residuals = copy(y)
	else
		parest = Xfact\y
		residuals = y - X*parest
	end
	return LinearRegression(y, X, parest, Xfact)
end
function regress(y::StridedVector{Float64}, X::StridedMatrix{Float64}, H::Matrix, h::Vector)
	fit = regress(y - X*h, X*H)
	fit.parest = H*fit.parest + h
	return fit
end

## Other methods

# ECM type
type ECM <: UnivariateRegressionModel
	endogenous::Vector{Float64}
	exogenous::Matrix{Float64}
	lags::Int64
	parest::Vector{Float64}
	response::Vector{Float64}
	predictor::Matrix{Float64}
	predfact::QRPivoted{Float64}
end

## Constructors
function ecm(endogenous::Vector{Float64}, exogenous::Matrix{Float64}, lags::Integer)
	T = length(endogenous) - lags
	dimexo = size(exogenous, 2)
	if size(exogenous, 1) != T + lags error("Endogenous and exogenous variables must have same sample length") end
	response = Array(Float64, T)
	predictor = Array(Float64, T, lags + dimexo*(lags + 1))
	# z2 = Array(Float64, T, lags - 1)
	for i = 1:T
		response[i] = endogenous[i+lags] - endogenous[i+lags-1]
		predictor[i,1] = endogenous[i+lags-1]
		cl = 2
		for j = 1:lags - 1
			predictor[i,cl] = endogenous[i+lags-j] - endogenous[i+lags-j-1]
			cl += 1
		end
		if dimexo > 0
			for k = 1:dimexo
				predictor[i,cl] = exogenous[i+lags-1,k]
				cl += 1
			end
			for j = 0:lags - 1
				for k = 1:dimexo
					predictor[i,cl] = exogenous[i+lags-j,k] - exogenous[i+lags-j-1,k]
					cl += 1
				end
			end
		end
	end
	fit = regress(response, predictor)
	return ECM(endogenous, exogenous, lags, fit.parest, response, predictor, fit.predfact)
end
ecm(endogenous::Vector{Float64}, lags::Integer) = ecm(endogenous, zeros(length(endogenous), 0), lags)

## Other methods

function show(io::IO, obj::ECM)
	println("Estimates t-stat")
	println(coeftable(obj))
end

function adf(fitA::ECM)
	iT 		= size(fitA.endogenous, 1) - 1
	kexo 	= size(fitA.exogenous, 2)
	ddendo 	= Array(Float64, iT)
	ddexo 	= Array(Float64, iT, kexo)
	for i = 1:iT
		ddendo[i] = fitA.endogenous[i + 1] - fitA.endogenous[i]
	end
	for j = 1:kexo
		for i = 1:iT
			ddexo[i,j] = fitA.exogenous[i + 1,j] - fitA.exogenous[i,j]
		end
	end
	fit0 	= adl(ddendo, ddexo, fitA.lags - 1)
	return 2*(loglikelihood(fitA) - loglikelihood(fit0))
end

# ECMI2 type
type ECMI2 <: UnivariateRegressionModel
	endogenous::Vector{Float64}
	exogenous::Matrix{Float64}
	lags::Int64
	parest::Vector{Float64}
	response::Vector{Float64}
	predictor::Matrix{Float64}
	predfact::QRPivoted{Float64}
end

## Constructors
function ecmI2(endogenous::Vector{Float64}, exogenous::Matrix{Float64}, lags::Integer)
	T = length(endogenous) - lags
	dimexo = size(exogenous, 2)
	if size(exogenous, 1) != T + lags error("Endogenous and exogenous variables must have same sample length") end
	response = Array(Float64, T)
	predictor = Array(Float64, T, lags + dimexo*(lags + 1))
	# z2 = Array(Float64, T, lags - 1)
	for i = 1:T
		response[i] = endogenous[i+lags] - 2endogenous[i+lags-1] + endogenous[i+lags-2]
		predictor[i,1] = endogenous[i+lags-1]
		predictor[i,2] = endogenous[i+lags-1] - endogenous[i+lags-2]
		cl = 3
		for j = 1:lags - 2
			predictor[i,cl] = endogenous[i+lags-j] - 2endogenous[i+lags-j-1] + endogenous[i+lags-j-2]
			cl += 1
		end
		if dimexo > 0
			for k = 1:dimexo
				predictor[i,cl] = exogenous[i+lags-1,k]
				cl += 1
			end
			for k = 1:dimexo
				predictor[i,cl] = exogenous[i+lags-1,k] - exogenous[i+lags-2,k]
				cl += 1
			end
			for j = 0:lags - 2
				for k = 1:dimexo
					predictor[i,cl] = exogenous[i+lags-j,k] - 2exogenous[i+lags-j-1,k] + exogenous[i+lags-j-1,k]
					cl += 1
				end
			end
		end
	end
	fit = regress(response, predictor)
	return ECMI2(endogenous, exogenous, lags, fit.parest, response, predictor, fit.predfact)
end
ecmI2(endogenous::Vector{Float64}, lags::Integer) = ecmI2(endogenous, zeros(length(endogenous), 0), lags)

## Other methods

function show(io::IO, obj::ECMI2)
	println("Estimates t-stat")
	println(coeftable(obj))
end

function adf(fitA::ECMI2)
	iT 		= size(fitA.endogenous, 1) - 2
	kexo 	= size(fitA.exogenous, 2)
	ddendo 	= Array(Float64, iT)
	ddexo 	= Array(Float64, iT, kexo)
	for i = 1:iT
		ddendo[i] = fitA.endogenous[i + 2] - 2fitA.endogenous[i + 1] + fitA.endogenous[i]
	end
	for j = 1:kexo
		for i = 1:iT
			ddexo[i,j] = fitA.exogenous[i + 2,j] - 2fitA.exogenous[i + 1,j] + fitA.exogenous[i,j]
		end
	end
	fit0 	= adl(ddendo, ddexo, fitA.lags - 2)
	return 2*(loglikelihood(fitA) - loglikelihood(fit0))
end
end #module