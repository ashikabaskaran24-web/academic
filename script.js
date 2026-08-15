function showSection(id) {
  document.querySelectorAll('.section').forEach(sec => {
    sec.classList.remove('active');
  });
  document.getElementById(id).classList.add('active');
}

// Auto-open detection page after prediction
window.onload = function () {
  if (document.querySelector('.result-box')) {
    showSection('detect');
  }
};

function translateTamil() {
  document.getElementById('tamilResult').style.display = 'block';
}
