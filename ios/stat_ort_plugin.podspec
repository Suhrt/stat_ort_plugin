#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint stat_ort_plugin.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'stat_ort_plugin'
  s.version          = '0.0.1'
  s.summary          = 'On-device Vaani ASR for Flutter via FFI (ONNX Runtime).'
  s.description      = <<-DESC
On-device speech recognition (Vaani ASR) for Flutter via dart:ffi — file and
streaming transcription with Silero VAD and speaker diarization, powered by
ONNX Runtime.
                       DESC
  s.homepage         = 'https://github.com/your-org/stat_ort_plugin'
  s.license          = { :type => 'MIT', :file => '../LICENSE' }
  s.author           = { 'stat_ort_plugin contributors' => 'email@example.com' }

  # This will ensure the source files in Classes/ are included in the native
  # builds of apps using this FFI plugin. Podspec does not support relative
  # paths, so Classes contains a forwarder C file that relatively imports
  # `../src/*` so that the C sources can be shared among all target platforms.
  s.source           = { :path => '.' }
  s.source_files = 'Classes/**/*'
  s.dependency 'Flutter'
  s.platform = :ios, '16.0'

  # Flutter.framework does not contain a i386 slice.
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386' }
  s.swift_version = '5.0'

  #added code
  s.dependency 'onnxruntime-c', '1.17.1'
  s.pod_target_xcconfig = { 'ENABLE_BITCODE' => 'NO' }
  s.static_framework = true
  # Force the linker to keep the plugin's exported symbols. The app only
  # resolves them at runtime via DynamicLibrary.process(), so without an
  # anchor they'd be dead-stripped. Naming one FFI_EXPORT function is enough:
  # iOS compiles all sources into a single unit (Classes/stat_ort_plugin.c),
  # so anchoring one symbol keeps them all.
  s.user_target_xcconfig = {
      'OTHER_LDFLAGS' => '-Wl,-u,_vaani_pipeline_init'
    }
end
