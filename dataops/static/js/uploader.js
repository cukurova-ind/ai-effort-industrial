document.getElementById("filepicker").addEventListener(
    "change",
    (event) => {
      let output = document.getElementById("listing");
      for (const file of event.target.files) {
        let item = document.createElement("li");
        item.textContent = file.webkitRelativePath;
        output.appendChild(item);
      }
    },
    false,
  );