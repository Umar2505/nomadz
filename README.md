# Nomadz
## Multi-Agent Flow:
* Agent 1: Interpret query -> make rules from pdf.
* Agent 2: Check compliance -> change input based on rules
* Agent 3: Retrieve/transform data -> 
* Agent 4: Generate a response
* Agent 5: Last check -> data anonymiser  

Transparency: Show users what privacy measures were applied and why.

## Project diagram
WIP

## 🔧 Firebase Setup

To enable Firebase in this project, follow these steps:

1. **Create a Firebase project**  
   Go to [Firebase Console](https://console.firebase.google.com/), sign in with your Google account, and create a new project.

2. **Retrieve your Firebase config**  
   In the Firebase Console, navigate to:  
   **Project Settings → General → Your Apps → Firebase SDK snippet (Config)**.  
   Copy the values provided.

3. **Add environment variables**  
   Create a `.env.local` file in the root of your project and add the following keys with your Firebase configuration values:

   ```env
   NEXT_PUBLIC_FIREBASE_API_KEY=your_api_key
   NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your_project.firebaseapp.com
   NEXT_PUBLIC_FIREBASE_PROJECT_ID=your_project_id
   NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=your_project.appspot.com
   NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=your_sender_id
   NEXT_PUBLIC_FIREBASE_APP_ID=your_app_id
   NEXT_PUBLIC_FIREBASE_MEASUREMENT_ID=your_measurement_id
