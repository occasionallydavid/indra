macro_rules! imported_model {
    ($name:ident) => {
        pub mod $name {
            #![allow(warnings, clippy::all)]
            include!(concat!(
                env!("OUT_DIR"),
                "/models/",
                stringify!($name),
                ".rs"
            ));
        }
    };
}

imported_model!(det);
imported_model!(rec);
imported_model!(order);
imported_model!(orient);
#[cfg(feature = "cjk")]
imported_model!(script);
#[cfg(feature = "cjk")]
imported_model!(rec_cjk);
#[cfg(feature = "math")]
imported_model!(mathcls);
#[cfg(feature = "math")]
imported_model!(math_enc);
#[cfg(feature = "math")]
imported_model!(math_dec_init);
#[cfg(feature = "math")]
imported_model!(math_dec_step);
