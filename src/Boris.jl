struct ElectrostaticBoris
  t::SVector{3, Float64}
  t²::Float64
  dt_2::Float64
end
function ElectrostaticBoris(B::AbstractVector, dt::Float64)
  t = (@SArray [B[1], B[2], B[3]]) * dt / 2
  t² = dot(t, t)
  return ElectrostaticBoris(t, t², dt / 2)
end
function (boris::ElectrostaticBoris)(vx, vy, vz, Ex, Ey, q_m)
  invγ = sqrt(1 - vx^2 - vy^2 - vz^2)
  q_mγ = q_m * invγ
  Ē₂ = (@SArray [Ex, Ey, 0.0]) * boris.dt_2 * q_mγ
  v⁻ = (@SArray [vx, vy, vz]) + Ē₂
  v⁺ = v⁻ + cross(v⁻ + cross(v⁻, boris.t), boris.t) * q_mγ^2 * 2 / (1 + q_mγ^2 * boris.t²)
  return (v⁺ + Ē₂) / invγ
end

struct ElectromagneticBoris
  dt_2::Float64
  ElectromagneticBoris(dt::Float64) = new(dt / 2)
end

function (boris::ElectromagneticBoris)(vx, vy, vz, Ex, Ey, Ez, Bx, By, Bz, q_m)
  invγ = sqrt(1 - vx^2 - vy^2 - vz^2)
  θ = boris.dt_2 * q_m * invγ
  t = (@SArray [Bx, By, Bz]) * θ
  tscale = 2 / (1 + dot(t, t))
  Ē₂ = (@SArray [Ex, Ey, Ez]) * θ
  v⁻ = (@SArray [vx, vy, vz]) + Ē₂
  v⁺ = v⁻ + cross(v⁻ + cross(v⁻, t), t) * tscale
  return (v⁺ + Ē₂) / invγ
end

