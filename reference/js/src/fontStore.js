export function initFontDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("CoursePdfFonts", 1);
    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains("fonts")) {
        db.createObjectStore("fonts", { keyPath: "id" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function saveCustomFont(name, arrayBuffer) {
  const db = await initFontDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction("fonts", "readwrite");
    const store = transaction.objectStore("fonts");
    const request = store.put({ id: "custom-font", name, data: arrayBuffer });
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

export async function loadCustomFont() {
  const db = await initFontDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction("fonts", "readonly");
    const store = transaction.objectStore("fonts");
    const request = store.get("custom-font");
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function deleteCustomFont() {
  const db = await initFontDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction("fonts", "readwrite");
    const store = transaction.objectStore("fonts");
    const request = store.delete("custom-font");
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}
