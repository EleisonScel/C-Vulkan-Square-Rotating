#version 450

layout (location = 0) in vec3 input_color_fragment;
layout (location = 0) out vec4 output_color;

void main() {
	output_color = vec4( input_color_fragment, 1.0 );
}
