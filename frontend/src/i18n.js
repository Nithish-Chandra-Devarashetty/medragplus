// Small set of UI strings. Medical content comes from the backend already translated.

export const LANGUAGES = [
  { code: 'auto', label: 'Auto' },
  { code: 'en', label: 'English' },
  { code: 'hi', label: 'हिन्दी' },
  { code: 'te', label: 'తెలుగు' },
]

export const SPEECH_LANG = { en: 'en-IN', hi: 'hi-IN', te: 'te-IN' }

const STRINGS = {
  en: {
    placeholder: 'Describe your symptoms…',
    send: 'Send',
    thinking: 'Thinking…',
    recording: 'Recording…',
    stop: 'Stop',
    newChat: 'New chat',
    history: 'Conversations',
    noSessions: 'No conversations yet.',
    autoSpeak: 'Auto-speak replies',
    sources: 'Sources',
    safety: 'Safety details',
    confidence: 'Confidence',
    logout: 'Log out',
    welcome: 'How are you feeling today?',
    welcomeSub: 'Describe your symptoms in English, Hindi or Telugu — type or use the microphone.',
    disclaimer: 'MedRAG+ provides triage guidance, not a medical diagnosis. In an emergency call 108 / 112.',
    escalated: 'Please seek medical help',
    emergencyBanner: 'Possible emergency — call 108 / 112 now',
  },
  hi: {
    placeholder: 'अपने लक्षण बताइए…',
    send: 'भेजें',
    thinking: 'सोच रहा है…',
    recording: 'रिकॉर्डिंग…',
    stop: 'रोकें',
    newChat: 'नई चैट',
    history: 'बातचीत',
    noSessions: 'अभी कोई बातचीत नहीं।',
    autoSpeak: 'जवाब बोलकर सुनाएँ',
    sources: 'स्रोत',
    safety: 'सुरक्षा विवरण',
    confidence: 'विश्वास',
    logout: 'लॉग आउट',
    welcome: 'आज आप कैसा महसूस कर रहे हैं?',
    welcomeSub: 'अपने लक्षण हिन्दी, अंग्रेज़ी या तेलुगु में लिखें या माइक का उपयोग करें।',
    disclaimer: 'MedRAG+ ट्राइएज मार्गदर्शन देता है, चिकित्सा निदान नहीं। आपात स्थिति में 108 / 112 पर कॉल करें।',
    escalated: 'कृपया चिकित्सा सहायता लें',
    emergencyBanner: 'संभावित आपात स्थिति — अभी 108 / 112 पर कॉल करें',
  },
  te: {
    placeholder: 'మీ లక్షణాలను వివరించండి…',
    send: 'పంపు',
    thinking: 'ఆలోచిస్తోంది…',
    recording: 'రికార్డింగ్…',
    stop: 'ఆపు',
    newChat: 'కొత్త చాట్',
    history: 'సంభాషణలు',
    noSessions: 'ఇంకా సంభాషణలు లేవు.',
    autoSpeak: 'సమాధానాలను వినిపించు',
    sources: 'మూలాలు',
    safety: 'భద్రతా వివరాలు',
    confidence: 'నమ్మకం',
    logout: 'లాగ్ అవుట్',
    welcome: 'ఈరోజు మీకు ఎలా ఉంది?',
    welcomeSub: 'మీ లక్షణాలను తెలుగు, హిందీ లేదా ఇంగ్లీష్‌లో టైప్ చేయండి లేదా మైక్ ఉపయోగించండి.',
    disclaimer: 'MedRAG+ ట్రయాజ్ మార్గదర్శనం మాత్రమే ఇస్తుంది, వైద్య నిర్ధారణ కాదు. అత్యవసర పరిస్థితిలో 108 / 112 కు కాల్ చేయండి.',
    escalated: 'దయచేసి వైద్య సహాయం పొందండి',
    emergencyBanner: 'అత్యవసర పరిస్థితి కావచ్చు — వెంటనే 108 / 112 కు కాల్ చేయండి',
  },
}

export function t(lang, key) {
  const table = STRINGS[lang] || STRINGS.en
  return table[key] ?? STRINGS.en[key] ?? key
}
