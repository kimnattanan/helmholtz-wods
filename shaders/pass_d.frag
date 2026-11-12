#version 450
//! Pass D stub: height = Re(u * e^{-iωt}); vsync off at present time.

layout(location = 0) out vec4 out_color;

layout(binding = 0, rg32f) uniform readonly image2D field_image;

layout(push_constant) uniform Push {
    vec2 resolution;
    float phase;
    float _pad;
} pc;

void main() {
    ivec2 pix = ivec2(gl_FragCoord.xy);
    vec2 u = imageLoad(field_image, pix).rg;
    float c = cos(pc.phase);
    float s = sin(pc.phase);
    float height = u.x * c + u.y * s; // Re(u * e^{-iφ}) with φ=phase
    float shade = height * 0.5 + 0.5;
    out_color = vec4(vec3(shade), 1.0);
}
