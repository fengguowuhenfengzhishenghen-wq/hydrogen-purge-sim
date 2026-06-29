# Task1 model interpretation

The transported variables are mole fractions. The dispersion model is K = beta * u * D + D_mol. At equal replacement progress, the interface variance scales as 2 K L / u, so velocity cancels when K is proportional to uD. Therefore mixed length is controlled mainly by pipe diameter and beta, not by displacement velocity or outlet back pressure.

Outlet back pressure affects density, Reynolds number, density Froude number diagnostics, and estimated frictional pressure drop. It does not change the mole-fraction profiles in this V1 constant-u model; this limitation must be stated in the report.

Recommended speed is selected at p_back=0.10 MPa by applying a hard dp/p_back < 0.10 feasibility constraint, then choosing the highest Fr among feasible speeds. This treats pressure drop as an engineering constraint and Fr as the primary stratification-risk screen.

The final mixed length and minimum effective N2 at 1.2 L/u may be zero because the pipe has already been flushed by nearly pure H2. For comparison, the summary CSV also reports values when the H2 front reaches 80% of pipe length and when H2 first breaks through the outlet.

Maximum dp/p_back in the sweep: 0.155. Recommended speeds at p_back=0.10 MPa: [{'D_m': 0.7, 'u_mps': 7.0}, {'D_m': 1.0, 'u_mps': 9.0}, {'D_m': 1.2, 'u_mps': 9.0}, {'D_m': 1.4, 'u_mps': 9.0}].
