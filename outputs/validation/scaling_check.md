# Austin-Palfrey scaling check

Austin-Palfrey style relations are treated only as an engineering order-of-magnitude cross-check here. They were developed for liquid sequential transport interface growth, while this project models weakly compressible gases with strong H2/air density contrast and possible horizontal-pipe stratification. Therefore the constants are not directly transferred into the gas purge model or used as strict validation.

The implemented beta_uD model should be read as a Taylor-Aris/Austin-Palfrey inspired engineering closure. The default beta=0.5 is deliberately conservative; outputs include beta=0.2/0.5/0.8 sensitivity so the report does not rely on a single uncalibrated knob.
