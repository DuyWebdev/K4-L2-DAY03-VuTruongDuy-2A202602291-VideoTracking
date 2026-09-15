# Báo cáo Ngày 3 — Tracking Annotation

Họ tên / nhóm: `Vũ Trường Duy / Cá nhân`
Ngày: `15/09/2026`

---

## 1. Quá trình gán nhãn

| Mục | Giá trị |
| --- | --- |
| Công cụ | CVAT / khác: `CVAT` |
| Thời gian gán `clip_02` (warm-up) | `CẦN BỔ SUNG` phút |
| Thời gian gán `clip_01` | `CẦN BỔ SUNG` phút |
| Số track đã vẽ trong `clip_01` | `8` |
| Số keyframe trung bình mỗi track | `CẦN BỔ SUNG – file MOT không lưu số keyframe CVAT` |

`clip_01` có 601 bbox trên 190 frame và 8 track. Phân bố track: ID 1: frame 1–11 (11 bbox); ID 2: frame 1–190 (190 bbox); ID 3: frame 1–45 (45 bbox); ID 4: frame 52–151 (100 bbox); ID 5: frame 82–140 (59 bbox); ID 6: frame 84–157 (74 bbox); ID 7: frame 106–190 (85 bbox); ID 8: frame 135–171 (37 bbox).

Ba tình huống khó nhất khi gán clip này, và bạn xử lý thế nào:

1. `CẦN BỔ SUNG – tình huống thực tế khi annotation.`
2. `CẦN BỔ SUNG – tình huống thực tế khi annotation.`
3. `CẦN BỔ SUNG – tình huống thực tế khi annotation.`

---

## 2. Tự kiểm và kiểm chéo

Ba lượt tua bắt được gì (lượt 1 nhìn ID, lượt 2 frame đầu/cuối, lượt 3 frame giữa):

- Lượt 1: `Kiểm tra ID: clip_01 có 8 ID (1–8), không có ID switch khi so với gold; IDF1 = 0.9591.`
- Lượt 2: `Kiểm tra đầu/cuối track: không có missed GT track và không có fragmented GT track; có 3 ghost-pred theo ranh giới track (ID 6 ở frame 84–100, ID 4 ở 149–151, ID 8 ở 169–171).`
- Lượt 3: `Kiểm tra vùng giữa clip: các bbox có IoU thấp nhất tập trung quanh frame 88–106 và 101–106; ngoài ra frame 55 có IoU 0.540 cho GT track 4. Đây là các vùng cần chú ý về độ sát bbox.`

Kiểm chéo với: `Không áp dụng – làm cá nhân.`  
Số lỗi bạn tìm được trong bản của bạn ấy: `N/A`. Số lỗi bạn ấy tìm được trong bản của bạn: `N/A`.

Ca nào hai người quyết khác nhau, và luật nào còn thiếu trong `GUIDELINE_MINI.md`?

`Không áp dụng vì làm cá nhân. Tuy nhiên, GUIDELINE_MINI.md còn thiếu các quy tắc cụ thể về ngưỡng occlusion, trường hợp xe cắt nhau/chồng nhau, xe nhỏ/mờ và mật độ keyframe. Các trường này nên được chốt rõ để lần gán sau nhất quán hơn.`

---

## 3. Pre-gold lock và chấm trước/sau rework

| Evidence | Giá trị |
| --- | --- |
| SHA-256 từ `evidence/pre-gold/clip_01/manifest.json` | `CHƯA CÓ – repo/evidence được cung cấp không chứa manifest.json` |
| Thời điểm khóa | `CHƯA CÓ – không có timestamp trong dữ liệu được cung cấp` |
| Số row / frame / track trước khi mở reference | `CHƯA CÓ pre-gold/gt.txt; annotation hiện tại: 601 row / 190 frame / 8 track` |

| | HOTA | DetA | AssA | LocA | IDF1 | MOTA | MOTP | FP | FN | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bản pre-gold | — | — | — | — | — | — | — | — | — | — |
| Sau rework / nhãn hiện tại vs gold | 0.8202 | 0.7981 | 0.8452 | 0.8855 | 0.9591 | 0.9162 | 0.8725 | 38 | 10 | 0 |

Qua cổng (`IDF1 >= 0.80`, `MOTA >= 0.75`, `MOTP >= 0.70`): **có**

Sau khi đọc danh sách lỗi, bạn đã sửa cụ thể những gì? Ghi theo frame và ID:

| Loại lỗi | Frame | ID | Đã sửa thế nào |
| --- | --- | --- | --- |
| Bbox lệch nhẹ | 55 | 4 | `CHƯA CÓ evidence rework; frame này có IoU 0.540 với gold` |
| Bbox lệch nhẹ | 88–96 | 5 | `CHƯA CÓ evidence rework; nhiều frame có IoU thấp` |
| Bbox lệch nhẹ | 101–106 | 6 | `CHƯA CÓ evidence rework; nhiều frame có IoU thấp` |

---

## 4. Kết quả model: ByteTrack control vs ReID treatment

Cấu hình từ `outputs/model_run_config.json`:

| Mục | Giá trị |
| --- | --- |
| Python / ultralytics / torch / lap | `3.13.15 / 8.4.145 / 2.11.0+cpu / 0.5.13` |
| weights / hai tracker | `yolo26n.pt / bytetrack.yaml / BoT-SORT + ReID (botsort-reid.yaml)` |
| conf / IoU / imgsz / classes | `0.25 / 0.70 / 960 / [2, 5, 7] (car, bus, truck)` |
| device | `cpu` |

| So sánh | HOTA | DetA | AssA | LocA | IDF1 | MOTA | MOTP | FP | FN | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bạn vs gold | 0.8202 | 0.7981 | 0.8452 | 0.8855 | 0.9591 | 0.9162 | 0.8725 | 38 | 10 | 0 |
| ByteTrack control vs gold | 0.7085 | 0.6487 | 0.7761 | 0.8463 | 0.8746 | 0.7487 | 0.8226 | 88 | 54 | 2 |
| BoT-SORT + ReID vs gold | 0.7635 | 0.7110 | 0.8204 | 0.8721 | 0.9001 | 0.7923 | 0.8595 | 91 | 26 | 2 |
| ReID vs bạn | 0.7705 | 0.7101 | 0.8369 | 0.8938 | 0.8781 | 0.7488 | 0.8886 | 94 | 57 | 0 |

---

## 5. Phân tích — năm câu hỏi

**1. MOTA của bạn cao hơn hay thấp hơn IDF1? Nếu MOTA cao mà IDF1 thấp thì điều đó nói gì, và vì sao MOTA không phạt nặng lỗi ID?**

`MOTA của nhãn tay cao hơn IDF1: MOTA = 0.9162 và IDF1 = 0.9591; ở đây thực tế IDF1 còn cao hơn MOTA. Điều này cho thấy annotation của tôi có chất lượng identity rất tốt nhưng vẫn còn một số lỗi detection/bbox. MOTA chủ yếu tổng hợp FP, FN và IDSW; vì vậy một số sai khác về định danh có thể không làm MOTA giảm mạnh nếu chúng không tạo ra nhiều IDSW. Trong kết quả của tôi, IDSW = 0 nên MOTA bị ảnh hưởng chủ yếu bởi 38 FP và 10 FN.`

**2. ByteTrack control và BoT-SORT + ReID treatment khác nhau thế nào ở IDF1, AssA và IDSW? Dẫn một frame sequence để giải thích treatment tốt hơn, tệ hơn hoặc không đổi đáng kể. Nhắc rõ đây không cô lập causal effect của ReID vì hai tracker implementation khác.**

`ReID tốt hơn ByteTrack ở IDF1 (0.9001 so với 0.8746) và AssA (0.8204 so với 0.7761), cho thấy association nhìn chung tốt hơn. Tuy nhiên IDSW của hai phương pháp đều bằng 2, nên ReID không làm giảm số ID switch trong lần chạy này. Ví dụ quanh frame 87–94, ReID có ID switch của GT track 5 tại frame 87; ByteTrack có ID switch của GT track 5 tại frame 94. Ở phía khác, ByteTrack có thêm switch của GT track 4 tại frame 59, trong khi ReID không có switch tương ứng ở track này. Vì vậy ReID cải thiện association tổng thể nhưng không loại bỏ hoàn toàn IDSW. Đây là so sánh hệ thống, không thể kết luận chênh lệch hoàn toàn do ReID vì ByteTrack và BoT-SORT là hai implementation tracker khác nhau.`

**3. DetA, FP và FN đổi thế nào? Lỗi còn lại là detector hay association?**

`So với gold, ReID tăng DetA từ 0.6487 lên 0.7110, giảm FN từ 54 xuống 26 nhưng FP tăng nhẹ từ 88 lên 91. Điều này cho thấy treatment phát hiện/giữ được nhiều đối tượng hơn, nhưng cũng sinh thêm một số detection dư. Association cũng được cải thiện vì AssA tăng từ 0.7761 lên 0.8204. Vì vậy lỗi còn lại không chỉ là association: ReID vẫn có vấn đề detection/false positive, trong khi một phần lỗi identity đã được cải thiện.`

**4. Một chỗ bạn đúng và ReID sai (frame, ID, vì sao):**

`Frame 190, đối tượng tương ứng với annotation ID 2 của tôi (gold ID 1): bbox của tôi có kích thước 95.26×41.60 và khớp gold với IoU xấp xỉ 0.90; bbox ReID tương ứng (track 3) chỉ có kích thước 61.45×36.45, IoU với gold khoảng 0.59. Vì vậy ở frame này annotation của tôi sát reference hơn về localization. ID số 2/3 khác nhau chỉ là track ID của hai hệ thống, không nên dùng số ID tuyệt đối để kết luận sai đúng.`

**5. Một chỗ ReID làm bạn xem lại annotation (frame, ID, vì sao), hoặc lý do evidence cho thấy model sai:**

`Frame 140, annotation của tôi vẫn có ID 5 với bbox nhỏ ở mép trái (x = 0, y ≈ 320.5), trong khi gold không có track 5 ở frame này và ReID cũng không giữ detection tương ứng. Đây là evidence đáng để xem lại annotation: có khả năng tôi đã giữ bbox thêm quá lâu khi xe đã không còn thỏa điều kiện gán nhãn. Tuy nhiên không nên sửa chỉ vì model khác; gold/reference mới là căn cứ chính.`

---

## 6. Nếu phải gán thêm 10 clip nữa

`Tôi sẽ bổ sung vào GUIDELINE_MINI.md các luật cụ thể thay vì để dấu "...": (1) ngưỡng occlusion tối đa để giữ ID; (2) quy tắc khi xe bị che lâu hơn ngưỡng; (3) quy tắc khi hai xe cắt/chồng nhau; (4) tiêu chí bắt đầu/kết thúc track khi xe nhỏ hoặc ở rìa ảnh; (5) quy tắc riêng cho xe đang đỗ; và (6) vị trí/mật độ keyframe. Quy trình làm việc nên là gán xong toàn clip trước khi xem model, sau đó tự kiểm ID và đầu/cuối track, rồi mới dùng model/evaluation để tìm các frame đáng nghi. Không dùng output của model làm gold.`

---

## 7. Tệp đã nộp

- [x] `annotations/clip_01/gt.txt`
- [x] `annotations/clip_02/gt.txt` — chưa kiểm tra thấy trong các file được cung cấp
- [x] `evidence/pre-gold/clip_01/gt.txt` — chưa có
- [x] `evidence/pre-gold/clip_01/manifest.json` — chưa có
- [x ] `GUIDELINE_MINI.md` đã điền — hiện vẫn còn placeholder
- [x] `outputs/eval_vs_gold.json`
- [x] `outputs/model_bytetrack_clip_01.txt`
- [x] `outputs/model_reid_clip_01.txt`
- [x] `outputs/model_run_config.json`
- [x] `outputs/eval_bytetrack_vs_gold.json`
- [x] `outputs/eval_reid_vs_gold.json`
- [x] `outputs/eval_reid_vs_me.json`
- [ ] `reports/review_partner.md` — không áp dụng vì làm cá nhân
- [x] `reports/REPORT.md` — đang hoàn thiện

