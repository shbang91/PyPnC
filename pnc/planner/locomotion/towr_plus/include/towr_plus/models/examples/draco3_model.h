#pragma once

#include <towr_plus/models/composite_rigid_body_dynamics.h>
#include <towr_plus/models/endeffector_mappings.h>
#include <towr_plus/models/examples/atlas_composite_rigid_body_inertia.h>
#include <towr_plus/models/kinematic_model.h>

namespace towr_plus {

class Draco3KinematicModel : public KinematicModel {
public:
  Draco3KinematicModel() : KinematicModel(2) {
    const double x_nominal_b = 0.05;
    const double y_nominal_b = 0.1;
    const double z_nominal_b = -0.65;

    // foot_half_length_ = 0.115;
    // foot_half_width_ = 0.065;
    foot_half_length_ = 0.08;
    foot_half_width_ = 0.04;

    nominal_stance_.at(L) << x_nominal_b, y_nominal_b, z_nominal_b;
    nominal_stance_.at(R) << x_nominal_b, -y_nominal_b, z_nominal_b;

    max_dev_from_nominal_ << 0.18, 0.1, 0.05;
    min_dev_from_nominal_ << -0.18, -0.1, -0.01;
  }
};

class Draco3DynamicModel : public CompositeRigidBodyDynamics {
public:
  Draco3DynamicModel()
      : CompositeRigidBodyDynamics(
            36.5222, 2, std::make_shared<AtlasCompositeRigidBodyInertia>()) {}
};

} /* namespace towr_plus */
