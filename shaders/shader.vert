#version 450

layout (binding = 0) uniform UniformBufferObject {
	mat4 model;
	mat4 view_projection;
} uniform_buffer_object;

layout (location = 0) in vec2 input_position;
layout (location = 1) in vec3 input_color;

layout (location = 0) out vec3 output_color_fragment;

void main() {
	output_color_fragment	= input_color;
	gl_Position				=
		uniform_buffer_object.view_projection * uniform_buffer_object.model *
		vec4( input_position, 0.0, 1.0 );
}
