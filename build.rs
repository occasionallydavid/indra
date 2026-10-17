use burn_onnx::ext::{
    ArgType, CodegenContext, Node, NodeType, OpOverride, ProcessError, arg_to_ident, quote::quote,
};
use burn_onnx::{LoadStrategy, ModelGen};
use onnx_ir::node::clip::ClipInput;

/// burn-onnx 0.22 emits a mixed-type runtime Clip bound in `math_enc`.
/// Keep the generated graph native, but select the bound type from the input
/// tensor so Burn receives matching scalar types.
struct ClipOverride;

impl OpOverride for ClipOverride {
    fn target(&self) -> NodeType {
        NodeType::Clip
    }

    fn forward(
        &self,
        node: &Node,
        ctx: &mut CodegenContext<'_, '_>,
    ) -> Result<burn_onnx::ext::proc_macro2::TokenStream, ProcessError> {
        let Node::Clip(clip) = node else {
            return Err(ProcessError::Custom("expected Clip node".into()));
        };
        let dtype = clip.inputs[0].ty.elem_type();
        let float = !dtype.is_int() && !dtype.is_uint();
        let bound = |value: &Option<ClipInput>| match value {
            None => None,
            Some(ClipInput::Static(value)) if float => Some(quote! { #value }),
            Some(ClipInput::Static(value)) => {
                let value = *value as i64;
                Some(quote! { #value })
            }
            Some(ClipInput::Runtime(runtime)) => {
                let input = &clip.inputs[runtime.input_index];
                let name = arg_to_ident(input);
                match input.ty {
                    ArgType::ScalarNative(_) if float => Some(quote! { #name as f64 }),
                    ArgType::ScalarNative(_) => Some(quote! { #name as i64 }),
                    _ => None,
                }
            }
        };
        let input = ctx.arg(&clip.inputs[0]);
        let output = arg_to_ident(&clip.outputs[0]);
        match (bound(&clip.config.min), bound(&clip.config.max)) {
            (Some(min), Some(max)) => Ok(quote! { let #output = #input.clamp(#min, #max); }),
            (Some(min), None) => Ok(quote! { let #output = #input.clamp_min(#min); }),
            (None, Some(max)) => Ok(quote! { let #output = #input.clamp_max(#max); }),
            (None, None) => Ok(quote! { let #output = #input; }),
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=models");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CJK");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_MATH");
    let mut generator = ModelGen::new();
    generator
        .input("models/det.onnx")
        .input("models/rec.onnx")
        .input("models/order.onnx")
        .input("models/orient.onnx");
    if std::env::var_os("CARGO_FEATURE_CJK").is_some() {
        generator
            .input("models/script.onnx")
            .input("models/rec_cjk.onnx");
    }
    if std::env::var_os("CARGO_FEATURE_MATH").is_some() {
        generator
            .input("models/mathcls.onnx")
            .input("models/math_enc.onnx")
            .input("models/math_dec_init.onnx")
            .input("models/math_dec_step.onnx");
    }
    generator
        .load_strategy(LoadStrategy::Embedded)
        .register_op_override(ClipOverride)
        .out_dir("models/")
        .run_from_script();
}
