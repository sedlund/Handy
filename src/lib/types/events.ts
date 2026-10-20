export interface ModelStateEvent {
  event_type: string;
  model_id?: string;
  model_name?: string;
  error?: string;
}

export interface TranscriptionDeviceEvent {
  device_name: string;
}

export interface RecordingErrorEvent {
  error_type: string;
  detail?: string;
}
