function smoothinfourierspace!(a, ffthelper)
  a .*= ffthelper.smoothingkernel
end
function smooth!(a, ffthelper)
  ffthelper.pfft! * a
  smoothinfourierspace!(a, ffthelper)
  ffthelper.pifft! * a
  a .= real(a)
end
for funcname in (:smooth!, :smoothinfourierspace!)
  @eval function $funcname(a, b, c, d, ffthelper::FFTHelper)
    $funcname(a, ffthelper)
    $funcname(b, ffthelper)
    $funcname(c, ffthelper)
    $funcname(d, ffthelper)
  end
end

# ∂ ρ = - ∇⋅J
function chargeconservation!(ρ⁺, ρ⁻, Jx, Jy, ffthelper, dt)
  @. ρ⁺ = ρ⁻ - im * (ffthelper.kx * Jx + ffthelper.ky * Jy) * dt
end
function chargeconservation!(ρ, Jx, Jy, ffthelper, dt)
  @. ρ += - im * (ffthelper.kx * Jx + ffthelper.ky * Jy) * dt
#  return chargeconservation!(ρ, ρ, Jx, Jy, ffthelper, dt)
end

## -∇² lhs = rhs
function neglaplacesolve!(lhs, rhs, ffthelper)
  ffthelper.pfft! * rhs
  @threads for j in axes(lhs, 2)
    for i in axes(lhs, 1)
      lhs[i, j] = rhs[i, j] / ffthelper.k²[i, j]
    end
  end
  lhs[1, 1] = 0
  ffthelper.pifft! * rhs
  #ffthelper.pifft! * lhs
end

@inline denominator(::Explicit, dt², k²) = 1
@inline denominator(::Implicit, dt², k²) = 1 + dt² * k² / 2
@inline denominator(imex::ImEx, dt², k²) = 1 + dt² * k² * theta(imex) / 2
@inline numerator(::Explicit, dt², k²) = 2 - dt² * k²
@inline numerator(::Implicit, dt², k²) = 2
@inline numerator(imex::ImEx, dt², k²) = 2 - dt² * k² * (1 - theta(imex))
# ∇^2 f - ∂ₜ² f = - s
# -k² f - (f⁺ - 2f⁰ + f⁻)/Δt^2 = - s
# Explicit f⁺ = (2 - k²Δt^2)f⁰ - f⁻ + Δt^2 s
function lorenzgauge!(imex::AbstractImEx, xⁿ, xⁿ⁻¹, sⁿ, k², dt²)
  @threads for i in eachindex(xⁿ)
    num = numerator(imex, dt², k²[i])
    den = denominator(imex, dt², k²[i])
    xⁿ⁺¹ = (num * xⁿ[i] + dt² * sⁿ[i]) / den - xⁿ⁻¹[i]
    xⁿ⁻¹[i] = xⁿ[i]
    xⁿ[i] = xⁿ⁺¹
  end
end

function lorenzgauge!(imex::AbstractImEx, xⁿ⁺¹, xⁿ, xⁿ⁻¹, sⁿ, k², dt²)
  θ = theta(imex)
  @threads for i in eachindex(xⁿ)
    num = numerator(imex, dt², k²[i])
    den = denominator(imex, dt², k²[i])
    xⁿ⁺¹[i] = (num * xⁿ[i] + dt² * sⁿ[i]) / den - xⁿ⁻¹[i]
  end
end

function lorenzgauge!(fieldimex::AbstractImEx, xⁿ⁺¹, xⁿ, xⁿ⁻¹, sⁿ⁺¹, sⁿ, sⁿ⁻¹, k², dt², sourceimex=fieldimex)
  θ = theta(sourceimex)
  @threads for i in eachindex(xⁿ)
    num = numerator(fieldimex, dt², k²[i])
    den = denominator(fieldimex, dt², k²[i])
    #xⁿ⁺¹[i] = (num * xⁿ[i] + dt² * (θ/2 * sⁿ⁻¹[i] + (1 - θ) * sⁿ[i] + θ/2 * sⁿ⁺¹[i])) / den - xⁿ⁻¹[i]
    xⁿ⁺¹[i] = (num * xⁿ[i] + dt² * (θ/2 * sⁿ⁻¹[i] + (1 - θ) * sⁿ[i] + θ/2 * sⁿ⁺¹[i])) / den - xⁿ⁻¹[i]
  end
end


